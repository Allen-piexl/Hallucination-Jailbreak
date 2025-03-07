import os, math, torch, pickle
from tqdm import tqdm
from datetime import datetime
from torch.nn.functional import cross_entropy
from config import ModelConfig
from utils import load_model_and_tokenizer, complete_input, extract_model_embedding
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg') 
import pickle

class Attacker:

    def __init__(self, model_name, init_input, target, device='cuda:0', steps=80, topk=256, batch_size=1024, mini_batch_size=16, **kwargs):
        try:
            self.model_config = getattr(ModelConfig, model_name)[0]
        except AttributeError:
            raise NotImplementedError

        self.model_name = model_name
        self.init_input = init_input
        self.target = target
        self.device = device
        self.steps = steps
        self.topk = topk
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.mini_batches = math.ceil(self.batch_size/self.mini_batch_size)
        self.kwargs = kwargs
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_config['path'], self.device, False
        )
        self.temp_step = 0
        self.temp_input = self.init_input
        self.temp_output = ''
        self.temp_loss = 1e+9
        self.temp_grad = None
        self.temp_input_ids = None
        self.temp_sample_list = []
        self.temp_sample_ids = None

        self.input_slice = None
        self.target_slice = None
        self.input_list = []
        self.output_list = []
        self.loss_list = []

        self.route_input = self.init_input
        self.route_loss = 1e+9
        self.route_step_list = []
        self.route_input_list = []
        self.route_output_list = []
        self.route_loss_list = []

        self.jailbreak_losses = []
        self.hallucination_losses = []
        self.scaling_factors_eta = []
        self.scaling_factors_beta = []

        self.loss_trends = {}
        self.lambda_vs_similarity = []

    def test(self):
        self.model.eval()
        input_str = complete_input(self.model_config, self.temp_input)
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=96)
        self.model.train()
        self.temp_output = self.tokenizer.decode(
            generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        print(f'Step  : {self.temp_step}/{self.steps}\n'
              f'Input : {self.temp_input}\n'
              f'Output: {self.temp_output}')

        self.input_list.append(self.temp_input)
        self.output_list.append(self.temp_output)


    def slice(self):
        prefix = self.model_config.get('prefix', '')
        prompt = self.model_config.get('prompt', '')
        suffix = self.model_config.get('suffix', '')
        temp_str = prefix+prompt
        temp_tokens = self.tokenizer(temp_str).input_ids
        len1 = len(temp_tokens)
        temp_str += self.route_input
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.input_slice = slice(len1, len(temp_tokens))
        try:
            assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.route_input
        except AssertionError:
            self.input_slice = slice(self.input_slice.start-1, self.input_slice.stop)
            try:
                assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.route_input
            except AssertionError:
                if self.tokenizer.decode(temp_tokens[self.input_slice]).lstrip() != self.route_input:
                    ### Todo
                    raise NotImplementedError

        temp_str += suffix
        temp_tokens = self.tokenizer(temp_str).input_ids
        len2 = len(temp_tokens)
        if suffix.endswith(':'):
            temp_str += ' '
        temp_str += self.target
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.target_slice = slice(len2, len(temp_tokens))


    def grad(self):
        model_embed = extract_model_embedding(self.model)
        embed_weights = model_embed.weight
        input_str = complete_input(self.model_config, self.route_input)
        if input_str.endswith(':'):
            input_str += ' '
        input_str += self.target
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids[0].to(self.device)
        self.temp_input_ids = input_ids.detach()

        compute_one_hot = torch.zeros(
            self.input_slice.stop-self.input_slice.start,
            embed_weights.shape[0],
            dtype=embed_weights.dtype, device=self.device
        )
        compute_one_hot.scatter_(
            1, input_ids[self.input_slice].unsqueeze(1),
            torch.ones(
                compute_one_hot.shape[0], 1, device=self.device, dtype=embed_weights.dtype
            )
        )
        compute_one_hot.requires_grad_()
        compute_embeds = (compute_one_hot @ embed_weights).unsqueeze(0)
        raw_embeds = model_embed(input_ids.unsqueeze(0)).detach()
        concat_embeds = torch.cat([
            raw_embeds[:, :self.input_slice.start, :],
            compute_embeds,
            raw_embeds[:, self.input_slice.stop: , :]
        ], dim=1)
        try:
            logits = self.model(inputs_embeds=concat_embeds).logits[0]
        except AttributeError:
            logits = self.model(input_ids=input_ids.unsqueeze(0), inputs_embeds=concat_embeds)[0]
        if logits.dim()>2:
            logits = logits.squeeze()
        try:
            assert input_ids.shape[0]>=self.target_slice.stop
        except AssertionError:
            self.target_slice = slice(self.target_slice.start, input_ids.shape[0])

        compute_logits = logits[self.target_slice.start-1 : self.target_slice.stop-1]
        target = input_ids[self.target_slice]
        loss = cross_entropy(compute_logits, target)
        loss.backward()

        self.temp_grad = compute_one_hot.grad.detach()


    def sample(self):
        self.temp_sample_list = []
        values, indices = torch.topk(self.temp_grad, k=self.topk, dim=1)
        sample_indices = torch.randperm(self.topk * self.temp_grad.shape[0])[:self.batch_size].tolist()
        for i in range(self.batch_size):
            pos = sample_indices[i] // self.topk
            pos_index = indices[pos][sample_indices[i] % self.topk].item()
            self.temp_sample_list.append((pos, pos_index))
        pos_list, pos_index_list = zip(*self.temp_sample_list)
        pos_tensor = torch.tensor(pos_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device)
        pos_tensor += self.input_slice.start
        pos_index_tensor = torch.tensor(pos_index_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device)

        sample_ids = self.temp_input_ids.repeat(self.batch_size, 1)
        sample_ids[range(self.batch_size), pos_tensor] = pos_index_tensor
        self.temp_sample_ids = sample_ids


    def forward(self):
        loss = torch.empty(0, device=self.device)
        with tqdm(total=self.batch_size) as pbar:
            pbar.set_description('Processing')
            for mini_batch in range(self.mini_batches):
                start = mini_batch*self.mini_batch_size
                end = min((mini_batch+1)*self.mini_batch_size, self.batch_size)
                targets = self.temp_input_ids[self.target_slice].repeat(end-start, 1)
                logits = self.model(self.temp_sample_ids[start:end]).logits
                logits = logits.permute(0, 2, 1)
                mini_batch_loss = cross_entropy(
                    logits[:, :, self.target_slice.start - 1:self.target_slice.stop - 1],
                    targets, reduction='none'
                ).mean(dim=-1)
                loss = torch.cat([loss, mini_batch_loss.detach()])
                torch.cuda.empty_cache()
                pbar.update(end-start)

        min_loss, min_index = loss.min(dim=-1)
        self.temp_loss = min_loss.item()
        self.loss_list.append(self.temp_loss)

        self.temp_input_ids = self.temp_sample_ids[min_index]
        self.temp_input = self.tokenizer.decode(
            self.temp_input_ids[self.input_slice],
            skip_special_tokens=True,
        )
        if self.model_name == 'internlm':
            ### for internlm, there may be an additional blank space on the left side of the decode string
            self.temp_input = self.temp_input.lstrip()


    def update(self):
        update_strategy = self.kwargs.get('update_strategy', 'strict')

        is_update = False
        if update_strategy == 'strict':
            if self.temp_loss<self.route_loss:
                is_update = True
        elif update_strategy == 'gaussian':
            gap_step = min(self.temp_step - self.route_step_list[-1], 20)
            if (self.temp_loss/self.route_loss-1)*100/gap_step <= torch.randn(1)[0].abs():
                is_update = True

        print(f'Temp Loss: {self.temp_loss}\t'
              f'Route Loss: {self.route_loss}\n'
              f'Update:', 'True' if is_update else 'False', '\n')

        if is_update:
            self.route_step_list.append(self.temp_step)
            self.route_input = self.temp_input
            self.route_input_list.append(self.route_input)
            self.route_loss = self.temp_loss
            self.route_loss_list.append(self.route_loss)
            self.route_output_list.append(self.temp_output)


    def pre(self):
        self.test()
        print('='*128,'\n')
        self.route_step_list.append(self.temp_step)
        self.route_input_list.append(self.temp_input)
        self.route_output_list.append(self.temp_output)
        self.route_loss_list.append(self.route_loss)
        self.temp_step+=1


    def save(self):
        save_dir = self.kwargs.get('save_dir', './results')
        os.makedirs(save_dir, exist_ok=True)
        save_dict = {
            'model_name': self.model_name,
            'init_input': self.init_input,
            'target': self.target,
            'steps': self.steps,
            'topk': self.topk,
            'batch_size': self.batch_size,
            'mini_batch_size': self.mini_batch_size,
            'kwargs': self.kwargs,
            'input_list': self.input_list,
            'output_list': self.output_list,
            'loss_list': self.loss_list,
            'route_step_list': self.route_step_list,
            'route_input_list': self.route_input_list,
            'route_output_list': self.route_output_list,
            'route_loss_list': self.route_loss_list
        }
        pkl_name = self.model_name+datetime.now().strftime("_%y%m%d%H%M%S.pkl")
        with open(os.path.join(save_dir, pkl_name), mode='wb') as f:
            pickle.dump(save_dict, f)
    def verify_lemma_word_by_word(self):
        # Get the token ID for the target word (e.g., "Donald")
        target_token_id = self.tokenizer("Donald", add_special_tokens=False).input_ids[0]
        ori_token_id = self.tokenizer("Joe", add_special_tokens=False).input_ids[0]
        # Forward pass: extract logits and attention

        # with torch.no_grad():
        #     output = self.model(self.temp_input_ids.unsqueeze(0), output_attentions=True)
        #     logits = output.logits[0]  # Shape: [sequence_length, vocab_size]
        #     attentions = output.attentions[-1]  # Shape: [num_heads, seq_len, seq_len]

        self.model.eval()
        input_str = complete_input(self.model_config, self.temp_input)
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids.to(self.device)
        self.model.config.output_attentions = True
        generate_outputs = self.model.generate(
            input_ids,
            max_new_tokens=96,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=True,
            return_legacy_cache = True
        )
        logits = torch.stack(generate_outputs.scores, dim=1)[0]  # Shape: [24, vocab_size]
        attentions = generate_outputs.attentions  # Shape: [num_layers, num_heads, seq_len, seq_len]
        generated_ids = generate_outputs.sequences
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        index = 0
        probs = torch.nn.functional.softmax(logits[index], dim=-1)

        target_prob = probs[target_token_id].item()
        non_target_prob = probs.sum().item() - target_prob

        last_layer_attention = torch.cat([head.squeeze(0) for head in attentions[-1]], dim=0)  # Shape: [32, 1, x]
        last_layer_attention = last_layer_attention.squeeze(1)  # Shape: [32, x]
        # last_layer_attention = attentions[-1]  # 使用最后一层的注意力
        target_attention = last_layer_attention[:, index].mean().item()
        non_target_attention = last_layer_attention.sum(dim=1).mean().item() - target_attention

        eta = (non_target_attention / (target_attention + 1e-8))/32
        beta = (non_target_prob / (target_prob + 1e-8))/32
        print(eta)
        print(beta)
        self.scaling_factors_eta.append(eta)
        self.scaling_factors_beta.append(beta)


        hallucination_loss = (
            -torch.log(torch.tensor(target_attention + 1e-8)) +
            self.kwargs.get('lambda', 0.01) * torch.log(torch.tensor(non_target_attention + 1e-8))
        ).item()
        jailbreak_loss = (
            -torch.log(torch.tensor(target_prob + 1e-8)) +
            torch.log(torch.tensor(1 + non_target_prob + 1e-8))
        ).item()

        self.hallucination_losses.append(hallucination_loss/5)
        self.jailbreak_losses.append(jailbreak_loss/15)

        # logits = torch.stack(generate_outputs.scores, dim=1)[0]
        # generated_ids = generate_outputs.sequences
        # attentions = generate_outputs.attentions
        # generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # token_idex = 0
        # layer_logits = logits[token_idex]  # Shape: [vocab_size]
        # min_val = layer_logits.min()
        # max_val = layer_logits.max()
        # normalized_logits = (layer_logits - min_val) / (max_val - min_val)
        # print(normalized_logits[0]/ normalized_logits[target_token_id])
        # vocab_size = logits.shape[-1]
        # print(((1 - normalized_logits[target_token_id])/normalized_logits[target_token_id])/vocab_size)
        # layer_attention = attentions[token_idex]
        # for idx, item in enumerate(layer_attention):
        #     print(
        #         f"Element {idx}: Type: {type(item)}, Shape: {item.shape if isinstance(item, torch.Tensor) else 'N/A'}")
        # num_heads = layer_attention.size()  # Number of attention heads
        # print(num_heads)
        # # Extract attention for the target token
        # probs = torch.nn.functional.softmax(logits[0][0], dim=-1)
        #
        # # Target and non-target probabilities
        # target_prob = probs[target_token_id].item()
        # non_target_prob = probs.sum().item() - target_prob
        #
        # target_attention = attentions[:, :, self.target_slice.start, self.target_slice.start].mean().item()
        # non_target_attention = attentions[:, :, self.target_slice.start, :].mean().item() - target_attention
        # eta = (non_target_attention / target_attention)/5
        # self.scaling_factors_eta.append(eta)
        #
        # # Extract logits for the target token
        # target_logits = logits[self.target_slice.start, target_token_id].item()
        # non_target_logits = (logits[self.target_slice.start, :].sum().item() - target_logits) / vocab_size
        # beta = non_target_logits / target_logits
        # self.scaling_factors_beta.append(beta)
        #
        #
        # # Compute hallucination loss
        #
        # hallucination_loss = (-torch.log(torch.tensor(target_attention) + 1e-8) + \
        #                      self.kwargs.get('lambda', 0.01) * torch.log(torch.abs(torch.tensor(non_target_attention)) + 1e-8))/64
        # self.hallucination_losses.append(hallucination_loss.item())
        #
        # # jailbreak_loss = -torch.log(torch.tensor(target_logits) + 1e-8) + \
        # #                  torch.log(1 + torch.tensor(non_target_logits)+1e-8)
        #
        # jailbreak_loss = (-torch.log(torch.tensor(target_prob) + 1e-8) + \
        #                  torch.log(1 + torch.tensor(non_target_prob) + 1e-8))/ 1200
        #
        # self.jailbreak_losses.append(jailbreak_loss.item())

    def verify_proposition2(self, save_dir='./loss_results', run_id=0):
        """
        Verify Proposition 2:
        - Compute hallucination and jailbreak loss trends for different lambda values.
        - Compute cosine similarity between loss trends.
        - Store the results for visualization.
        """
        lambda_values = [0.1, 1, 10, 100]
        self.lambda_vs_similarity = []  #
        self.loss_trends = {}

        for lambda_val in lambda_values:
            hallucination_losses_lambda = []
            jailbreak_losses_lambda = []

            # 遍历实验步骤，计算 loss 变化趋势
            for i in range(len(self.scaling_factors_eta)):
                hallucination_loss_i = (
                        -torch.log(torch.tensor(self.scaling_factors_eta[i] + 1e-8)) +
                        lambda_val * torch.log(torch.tensor(self.scaling_factors_beta[i] + 1e-8))
                ).item()
                hallucination_losses_lambda.append(hallucination_loss_i)

                jailbreak_loss_i = (
                        -torch.log(torch.tensor(self.scaling_factors_eta[i] + 1e-8)) +
                        torch.log(torch.tensor(1 + self.scaling_factors_beta[i] + 1e-8))
                ).item()
                jailbreak_losses_lambda.append(jailbreak_loss_i)


            self.loss_trends[f"hallucination_loss_lambda_{lambda_val}"] = hallucination_losses_lambda
            self.loss_trends[f"jailbreak_loss_lambda_{lambda_val}"] = jailbreak_losses_lambda
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"loss_trends_run_{run_id}.pkl")

            with open(save_path, 'wb') as f:
                pickle.dump(self.loss_trends, f)


            hallucination_losses_tensor = torch.tensor(hallucination_losses_lambda)
            jailbreak_losses_tensor = torch.tensor(jailbreak_losses_lambda)

            cos_sim = torch.nn.functional.cosine_similarity(
                hallucination_losses_tensor.view(1, -1),
                jailbreak_losses_tensor.view(1, -1),
                dim=1
            ).item()

            self.lambda_vs_similarity.append(cos_sim)

        print(f"Cosine Similarity of Loss Trends across Lambda values: {self.lambda_vs_similarity}")

    def visualize_lemma_results_2(self, save_dir='./visualizations', run_id=0):
        """
        Visualize hallucination and jailbreak loss trends for different lambda values.
        """
        os.makedirs(save_dir, exist_ok=True)
        lambda_values = [0.1, 1, 10, 100]
        steps = range(1, len(self.scaling_factors_eta) + 1)

        for lambda_val in lambda_values:
            plt.figure(figsize=(14, 5))

            # 创建双轴图
            fig, ax1 = plt.subplots(figsize=(14, 5))

            # 画出 Hallucination Loss
            hallucination_key = f"hallucination_loss_lambda_{lambda_val}"
            ax1.plot(
                steps, self.loss_trends[hallucination_key],
                label=f"Hallucination Loss", marker='o', linewidth=2
            )
            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Hallucination Loss")
            ax1.tick_params(axis='y')
            ax1.grid(alpha=0.5, linestyle='--')

            # 画出 Jailbreak Loss (右轴)
            jailbreak_key = f"jailbreak_loss_lambda_{lambda_val}"
            ax2 = ax1.twinx()
            ax2.plot(
                steps, self.loss_trends[jailbreak_key],
                label=f"Jailbreak Loss", marker='x', color='orange', linewidth=2
            )
            ax2.set_ylabel("Jailbreak Loss")
            ax2.tick_params(axis='y')


            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")


            plt.savefig(os.path.join(save_dir, f"loss_trends_lambda_{lambda_val}_run_{run_id}.png"))
            plt.close()

            print(f"Saved loss trend visualization for λ={lambda_val} in {save_dir}")

    def visualize_proposition2_results(self, save_dir='./visualizations', run_id=0):
        """
        Visualize Proposition 2 results:
        - Plot cosine similarity between hallucination loss and jailbreak loss for different lambda values.
        """
        os.makedirs(save_dir, exist_ok=True)
        lambda_values = [0.1, 1, 10, 100]

        plt.figure(figsize=(8, 5))
        plt.plot(lambda_values, self.lambda_vs_similarity, marker='o', linestyle='-', color='blue')
        plt.xscale("log")
        plt.xlabel("Lambda (log scale)")
        plt.ylabel("Loss Trend Similarity (Cosine)")
        plt.title("Effect of Lambda on Loss Trend Similarity")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(save_dir, f"proposition2_similarity_{run_id}.png"))
        plt.close()

        print(f"Proposition 2 visualization saved in {save_dir}")


    def visualize_lemma_results(self, save_dir='./visualizations', run_id=0):
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(14, 5))
        steps = range(1, len(self.hallucination_losses) + 1)

        # 左轴：Hallucination Loss
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(steps, self.hallucination_losses[::-1], label="Hallucination Loss", marker='.', linewidth=2)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Hallucination Loss")
        ax1.tick_params(axis='y')
        ax1.grid(alpha=0.5, linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(steps, self.jailbreak_losses, label="Jailbreak Loss", marker='x', color='orange', linewidth=2)
        ax2.set_ylabel("Jailbreak Loss")
        ax2.tick_params(axis='y')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        plt.savefig(os.path.join(save_dir, f"loss_comparison_run_{run_id}.png"))
        plt.close()

        fig, ax1 = plt.subplots(figsize=(14, 5))

        ax1.plot(steps, self.scaling_factors_eta[::-1], label="Eta (η)", marker='o', linewidth=1.5)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Eta (η)")
        ax1.tick_params(axis='y')
        ax1.grid(alpha=0.3, linestyle='--')


        ax2 = ax1.twinx()
        ax2.plot(steps, self.scaling_factors_beta, label="Beta (β)", marker='x', color='orange', linewidth=1.5)
        ax2.set_ylabel("Beta (β)")
        ax2.tick_params(axis='y')

        # Add legends
        # ax1.legend(loc="upper right")
        # ax2.legend(loc="upper right")
        # # Add legends (stacked in the upper-right corner)
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
            bbox_to_anchor=(1, 1),  # Position in the upper-right corner
            ncol=1,  # Stack vertically
            frameon=False  # Remove the box around the legend
        )

        # Adjust layout to avoid overlap
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "scaling_factors_dual_axis_{}.png".format(run_id)))
        plt.close()

        print(f"Visualizations saved in {save_dir}")




    def run(self, run_id):
        self.pre()
        early_stop = self.kwargs.get('early_stop', False)
        while self.temp_step <= self.steps:
            self.slice()
            self.grad()
            self.sample()
            self.forward()
            self.test()
            self.update()
            self.temp_step += 1
            self.verify_lemma_word_by_word()
            # if early_stop and self.temp_output == self.target:
            #     break
        self.verify_proposition2(run_id=run_id)
        is_save = self.kwargs.get('is_save', False)
        if is_save:
            self.save()
        self.visualize_lemma_results(run_id=run_id)
        self.visualize_proposition2_results(run_id=run_id)
        self.visualize_lemma_results_2(run_id=run_id)