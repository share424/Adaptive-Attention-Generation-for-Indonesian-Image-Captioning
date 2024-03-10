from typing import Union, Tuple, TypeVar
from queue import PriorityQueue
import operator
import math

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np

from image_captioning.models.model_registry import register_model


RNNState = TypeVar("RNNState", Tensor, tuple[Tensor, Tensor])


class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, states: tuple[Tensor, Tensor]) -> tuple[RNNState, Tensor]:
        """Forward pass
        
        Args:
            x (Tensor): input tensor (B, input_size)
            states (tuple[Tensor, Tensor]): hidden and cell states (B, hidden_size)

        Returns:
            tuple[Tensor, Tensor, Tensor]: hidden state, cell state, visual sentinel
        """
        h_old, m_old = states

        # do LSTM, and get new hidden and output
        ht, mt = self.lstm_cell(x, (h_old, m_old))

        # do sigmoid to the input and hidden to get visual sentinel St (Eq. 1 in the paper)
        gt = F.sigmoid(self.x_gate(x) + self.h_gate(h_old))

        # and then do tanh to get visual sentinel (Eq. 2 in the paper)
        st = gt * F.tanh(mt)

        return (ht, mt), st
    

class AdaptiveGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> tuple[RNNState, Tensor]:
        """Forward pass
        
        Args:
            x (Tensor): input tensor (B, input_size)
            h (Tensor): hidden state (B, hidden_size)

        Returns:
            tuple[Tensor, Tensor]: hidden state, visual sentinel
        """
        h_old = h

        # do GRU, and get new hidden and output
        ht = self.gru_cell(x, h_old)

        # do sigmoid to the input and hidden to get visual sentinel St (Eq. 1 in the paper)
        gt = F.sigmoid(self.x_gate(x) + self.h_gate(h_old))

        # and then do tanh to get visual sentinel (Eq. 2 in the paper)
        st = gt * F.tanh(ht)

        return ht, st


class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        self.sentinel_affine = nn.Linear(hidden_size, hidden_size)
        self.sentinel_attention = nn.Linear(hidden_size, attention_size)
        
        self.hidden_affine = nn.Linear(hidden_size, hidden_size)
        self.hidden_attention = nn.Linear(hidden_size, attention_size)
        self.visual_attention = nn.Linear(hidden_size, attention_size)
        self.alphas = nn.Linear(attention_size, 1)
        self.context_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, spatial_image: Tensor, decoder_output: Tensor, st: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass
        
        Args:
            spatial_image (Tensor): spatial image (B, num_pixels, hidden_size)
            decoder_output (Tensor): decoder output (B, hidden_size)
            st (Tensor): sentinel (B, hidden_size)

        Returns:
            tuple[Tensor, Tensor, Tensor]: output, attention_weights, beta_value (B, hidden_size), (B, num_pixels + 1), (B, 1)
        """
        num_pixels = spatial_image.shape[1]

        # get the visual attention using spatial_image as input
        visual_attention: Tensor = self.visual_attention(spatial_image) # (B, num_pixels, attention_size)

        # get sentinel affine using st as input with ReLU activation
        sentinel_affine = F.relu(self.sentinel_affine(st)) # (B, hidden_size)

        # get sentinel attention using sentinel_affine as input
        sentinel_attention: Tensor = self.sentinel_attention(sentinel_affine) # (B, attention_size)

        hidden_affine = F.tanh(self.hidden_affine(decoder_output)) # (B, hidden_size)
        hidden_attention: Tensor = self.hidden_attention(hidden_affine) # (B, attention_size)

        hidden_features = hidden_attention \
                .unsqueeze(1) \
                .expand(
                    hidden_attention.size(0), 
                    num_pixels + 1, 
                    hidden_attention.size(1)
                ) # (B, num_pixels + 1, attention_size)

        concat_features = torch.cat([spatial_image, sentinel_affine.unsqueeze(1)], dim=1) # (B, num_pixels + 1, hidden_size)
        attended_features = torch.cat([visual_attention, sentinel_attention.unsqueeze(1)], dim=1) # (B, num_pixels, attention_size)

        # do tanh to attended features and hidden features
        attention = F.tanh(attended_features + hidden_features) # (B, num_pixels + 1, attention_size)
        
        # do a forward linear layer to get the alphas
        alphas: Tensor = self.alphas(attention).squeeze(2) # (B, num_pixels + 1)

        # and softmax to get the attention weights (Eq. 5 in the paper)
        attention_weights = F.softmax(alphas, dim=1) # (B, num_pixels + 1)

        context = (concat_features * attention_weights.unsqueeze(2)).sum(dim=1) # (B, hidden_size)

        # get the new beta value by getting the last value of attention weights
        beta_value = attention_weights[:, -1].unsqueeze(1) # (B, 1)

        out = F.tanh(self.context_hidden(context + hidden_affine)) # (B, hidden_size)

        return out, attention_weights, beta_value
    

class AdaptiveAttentionRNN(nn.Module):
    rnn: nn.Module

    def __init__(
        self, 
        hidden_size: int, 
        vocab_size: int, 
        attention_size: int, 
        embedding_size: int, 
        encoded_dimension: int,
        start_token: int,
        end_token: int,
        dropout: float = 0.5,
        max_prediction_length: int = 20,
        search_strategy: str = "greedy_search",
        **kwargs
    ):
        super().__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.encoded_to_hidden = nn.Linear(encoded_dimension, hidden_size)
        self.global_features = nn.Linear(encoded_dimension, embedding_size)
        self.adaptive_attention = AdaptiveAttention(hidden_size, attention_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.max_prediction_length = max_prediction_length
        self.search_strategy = search_strategy
        self.kwargs = kwargs

        self._init_rnn()

    def _init_rnn(self):
        raise NotImplementedError

    def init_hidden_state(self, batch_size: int, device: str = "cpu"):
        raise NotImplementedError
    
    def _get_adaptive_attention_hidden_state(self, state: RNNState) -> Tensor:
        if isinstance(state, tuple):
            return state[0]
        
        return state

    def _decode(
        self,
        tokens: Tensor,
        global_image: Tensor,
        spatial_image: Tensor,
        state: RNNState,
    ) -> Tuple[Tensor, Tensor, Tensor, RNNState]:
        embeddings = self.embedding(tokens)
        inputs = torch.cat((embeddings, global_image.expand_as(embeddings)), dim=1)
        state, st = self.rnn(inputs, state)
        h = self._get_adaptive_attention_hidden_state(state)
        out_l, alpha, beta = self.adaptive_attention(spatial_image, h, st)
        prediction = self.fc(out_l)
        prediction = F.log_softmax(prediction, dim=1)

        return prediction, alpha, beta, state

    def _get_state_with_size(self, state: RNNState, size: int) -> RNNState:
        if isinstance(state, tuple):
            h, c = state
            return (h[:size], c[:size])
        
        return state[:size]

    def _get_state_by_index(self, state: RNNState, index: int) -> RNNState:
        if isinstance(state, tuple):
            h, c = state
            return (h[index].unsqueeze(0), c[index].unsqueeze(0))
        
        return state[index].unsqueeze(0)

    def _forward_train(
        self,
        encoded_image: Tensor,
        global_features: Tensor, 
        encoded_captions: Tensor,
        caption_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = encoded_image.size(0)
        num_pixels = encoded_image.size(1)

        spatial_image = F.relu(self.encoded_to_hidden(encoded_image))
        global_image = F.relu(self.global_features(global_features))

        # sort input data by decreasing length
        # caption_length will contains the sorted length, and sort_idx will contains the sorted elements indices
        caption_lengths, sort_idx = caption_lengths.squeeze(1).sort(dim=0, descending=True)

        # sort spatial_image, global_features, encoded_captions and encoded_image batches by caption length
        spatial_image = spatial_image[sort_idx]
        global_image = global_image[sort_idx]
        encoded_captions = encoded_captions[sort_idx]
        encoded_image = encoded_image[sort_idx]

        # Embedding, each batch contains a caption. All batch have the same number of rows (words), since we previously
        # padded the ones shorter than max_caption_lengths, as well as the same number of columns (embed_dimension)
        embeddings = self.embedding(encoded_captions) # (B, max_caption_length, embedding_size)

        # initialize the hidden state
        state = self.init_hidden_state(batch_size, encoded_image.device)

        # we won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to store word prediction score, alphas and betas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoded_image.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels + 1).to(encoded_image.device)
        betas = torch.zeros(batch_size, max(decode_lengths), 1).to(encoded_image.device)

        # concenate the embeddings and global image feature for RNN input
        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        inputs = torch.cat((global_image, embeddings), dim=2) # (B, max_caption_length, embedding_size*2)

        # start decoding
        for timestep in range(max(decode_lengths)):
            # create a packet sequence to process the only effective batch size N_t at that timestep
            batch_size_t = sum([l > timestep for l in decode_lengths])
            current_input = inputs[:batch_size_t, timestep, :] # (batch_size_t, embedding_size*2)

            # do RNN
            current_state = self._get_state_with_size(state, batch_size_t)
            state, st = self.rnn(current_input, current_state)

            # run the adaptive attention
            current_h = self._get_adaptive_attention_hidden_state(state)
            out_l, alpha_t, beta_t = self.adaptive_attention(spatial_image[:batch_size_t], current_h, st) # (batch_size_t, hidden_size), (batch_size_t, num_pixels + 1), (batch_size_t, 1)

            # compute the probability over the vocabulary with fully connected layer
            prediction = self.fc(self.dropout(out_l))

            # store the prediction, alphas and betas value
            predictions[:batch_size_t, timestep, :] = prediction
            alphas[:batch_size_t, timestep, :] = alpha_t
            betas[:batch_size_t, timestep, :] = beta_t

        # Since we decoded starting caption with <start> token, the targets are all words after <start> up to <end>
        targets = encoded_captions[:, 1:]
        
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        return scores, targets, decode_lengths
    
    def _forward_greedy_search(
        self,
        spatial_image: Tensor,
        global_image: Tensor,
        state: RNNState,
        **kwargs
    ):
        batch_size = spatial_image.size(0)

        # set first input as <start> token
        previous_word = [self.start_token] * batch_size
        previous_word = torch.LongTensor(previous_word).to(spatial_image.device)

        predictions = torch.zeros(batch_size, 1, self.max_prediction_length).to(spatial_image.device)
        alphas = torch.zeros(batch_size, 1, self.max_prediction_length, spatial_image.size(1) + 1).to(spatial_image.device)
        betas = torch.zeros(batch_size, 1, self.max_prediction_length, 1).to(spatial_image.device)
        
        # Inference
        for step in range(self.max_prediction_length):
            prediction, alpha, beta, state = self._decode(
                previous_word, global_image, spatial_image, state)

            # get the maximum value of the prediction
            _, predicted_word = prediction.max(1)

            # store the predicted word
            previous_word = predicted_word

            # store the prediction
            predictions[:, 0, step] = predicted_word
            alphas[:, 0, step, :] = alpha
            betas[:, 0, step, :] = beta

        return predictions.detach().cpu().numpy(), alphas.detach().cpu().numpy(), betas.detach().squeeze(-1).cpu().numpy()
    
    def _forward_beam_search(
        self,
        spatial_image: Tensor,
        global_image: Tensor,
        state: RNNState,
        **kwargs
    ):
        """Forward pass using beam search
        
        Args:
            spatial_image (Tensor): spatial image (B, num_pixels, hidden_size)
            global_image (Tensor): global image (B, embedding_size)
            state (RNNState): hidden state (B, hidden_size)

        Returns:
            tuple[Tensor, Tensor, Tensor]: decoded_batch, decoded_alphas, decoded_betas

        References:
            [1] https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
        """
        beam_size = kwargs.get("beam_size", self.kwargs.get("beam_size", 3))
        topk = kwargs.get("num_of_generated", self.kwargs.get("num_of_generated", 1))
        batch_size = spatial_image.size(0)
        pixel_size = spatial_image.size(1)
        decoded_batch = []
        decoded_alphas = []
        decoded_betas = []

        for idx in range(batch_size):
            initial_state = self._get_state_by_index(state, idx)

            decoder_input = torch.LongTensor([self.start_token]).to(spatial_image.device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            node = BeamSearchNode(
                state=initial_state,
                previous_node=None,
                token=decoder_input,
                alpha=torch.zeros((int(math.sqrt(pixel_size)), int(math.sqrt(pixel_size)))).tolist(),
                beta=0,
                log_prob=0,
                length=1
            )
            nodes = PriorityQueue()

            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                if qsize > 100: break

                score, node = nodes.get()
                decoder_input = node.token
                decoder_state = node.state

                if node.token.item() == self.end_token and node.previous_node is not None:
                    endnodes.append((score, node))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                predictions, alphas, betas, current_state = self._decode(
                    decoder_input, 
                    global_image[idx], 
                    spatial_image[idx].unsqueeze(0), 
                    decoder_state
                )
                alphas = alphas[:,:-1]
                alphas = alphas.view(-1, int(math.sqrt(pixel_size)), int(math.sqrt(pixel_size)))

                log_prob, indexes = torch.topk(predictions, beam_size)
                nextnodes = []

                for new_k in range(beam_size):
                    decoded_t = indexes[0][new_k].view(1, -1).squeeze(0)
                    alpha = alphas[0].detach().cpu().numpy().tolist()
                    beta = betas[0].item()
                    log_p = log_prob[0][new_k].item()

                    new_node = BeamSearchNode(
                        state=current_state,
                        previous_node=node,
                        token=decoded_t,
                        alpha=alpha,
                        beta=beta,
                        log_prob=node.log_prob + log_p,
                        length=node.length + 1
                    )
                    score = -new_node.eval()
                    nextnodes.append((score, new_node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                
                # increase qsize
                qsize += 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            alphas = []
            betas = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                alpha = []
                beta = []
                utterance.append(n.token.item())
                alpha.append(n.alpha)
                beta.append(n.beta)
                # back trace
                while n.previous_node != None:
                    n = n.previous_node
                    utterance.append(n.token.item())
                    alpha.append(n.alpha)
                    beta.append(n.beta)

                utterance = utterance[::-1]
                alpha = alpha[::-1]
                beta = beta[::-1]
                utterances.append(np.array(utterance))
                alphas.append(np.array(alpha))
                betas.append(np.array(beta))

            decoded_batch.append(utterances)
            decoded_alphas.append(alphas)
            decoded_betas.append(betas)

        return decoded_batch, decoded_alphas, decoded_betas

    def forward(
        self, 
        encoded_image: Tensor, 
        global_features: Tensor, 
        encoded_captions: Tensor = None, 
        caption_lengths: Tensor = None,
        **kwargs
    ):
        """Forward pass

        Args:
            encoded_image (Tensor): encoded image (B, num_pixels, hidden_size)
            global_features (Tensor): global features (B, embedding_size)
            encoded_captions (Tensor): encoded captions (B, max_caption_length)
            caption_lengths (Tensor): caption lengths (B,)

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: predictions, alphas, betas, encoded_captions, decode_lengths, sort_idx
        """
        if self.training:
            return self._forward_train(
                encoded_image,
                global_features, 
                encoded_captions,
                caption_lengths,
                **kwargs
            )
        
        batch_size = encoded_image.size(0)

        spatial_image = F.relu(self.encoded_to_hidden(encoded_image)) # (B, num_pixels, hidden_size)
        global_image = F.relu(self.global_features(global_features)) # (B, embedding_size)

        # Initialize the RNN state
        state = self.init_hidden_state(batch_size, encoded_image.device) # (B, hidden_size)

        
        search_strategy = kwargs.get("search_strategy", self.search_strategy)

        if search_strategy == "greedy_search":
            return self._forward_greedy_search(
                spatial_image,
                global_image,
                state,
                **kwargs
            )
        elif search_strategy == "beam_search":
            return self._forward_beam_search(
                spatial_image,
                global_image,
                state,
                **kwargs
            )
        # elif search_strategy == "beam_search_v2":
        #     return self._forward_beam_search_v2(
        #         encoded_image,
        #         global_image,
        #         **kwargs
        #     )
        else:
            raise ValueError(f"Unknown search method: {search_strategy}")

@register_model
class AdaptiveAttentionLSTM(AdaptiveAttentionRNN):
    def _init_rnn(self):
        self.rnn = AdaptiveLSTMCell(self.embedding.embedding_dim * 2, self.hidden_size)

    def init_hidden_state(self, batch_size: int, device: str = "cpu"):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return h, c 

@register_model
class AdaptiveAttentionGRU(AdaptiveAttentionRNN):
    def _init_rnn(self):
        self.rnn = AdaptiveGRUCell(self.embedding.embedding_dim * 2, self.hidden_size)

    def init_hidden_state(self, batch_size: int, device: str = "cpu"):
        return torch.zeros(batch_size, self.hidden_size).to(device)

class BeamSearchNode:
    def __init__(
        self, 
        state: RNNState,
        previous_node: "BeamSearchNode", 
        token: int,
        alpha: float,
        beta: float,
        log_prob: float, 
        length: int
    ):
        self.state = state
        self.previous_node = previous_node
        self.token = token
        self.alpha = alpha
        self.beta = beta
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward
    
    # define binary comparison for nodes
    def __lt__(self, other):
        if isinstance(other, BeamSearchNode):
            return -self.eval() < -other.eval()
        
        return -self.eval() < other
    
    def __le__(self, other):
        if isinstance(other, BeamSearchNode):
            return -self.eval() <= -other.eval()
        
        return -self.eval() <= other
    
    def __gt__(self, other):
        if isinstance(other, BeamSearchNode):
            return -self.eval() > -other.eval()
        
        return -self.eval() > other
    
    def __ge__(self, other):
        if isinstance(other, BeamSearchNode):
            return -self.eval() >= -other.eval()
        
        return -self.eval() >= other
    
    def __eq__(self, other):
        if isinstance(other, BeamSearchNode):
            return self.eval() == other.eval()
        
        return self.eval() == other
    
    def __ne__(self, other):
        if isinstance(other, BeamSearchNode):
            return self.eval() != other.eval()
        
        return self.eval() != other