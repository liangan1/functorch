from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, ReformerConfig, BigBirdConfig, BertConfig
import transformers
import torch
from functorch.compile import memory_efficient_fusion, aot_module, draw_graph_compile, nop
import torch.utils._pytree as pytree
import time
from torch import optim
import torch.nn as nn
from torch.nn.utils import _stateless
import pandas as pd
from functorch.compile import compiled_module, tvm_compile

pytree._register_pytree_node(transformers.modeling_outputs.MaskedLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.MaskedLMOutput(loss=values[0], logits=values[1]))

pytree._register_pytree_node(transformers.modeling_outputs.Seq2SeqLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.Seq2SeqLMOutput(loss=values[0], logits=values[1]))

pytree._register_pytree_node(transformers.modeling_outputs.CausalLMOutputWithCrossAttentions, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions(loss=values[0], logits=values[1]))

pytree._register_pytree_node(transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput(loss=values[0], logits=values[1]))

torch.manual_seed(42)
benchmarks = [
    #(AutoConfig.from_pretrained("albert-base-v2"), AutoModelForMaskedLM, (8, 512), []),
    #(AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (4, 512), []),
    (BertConfig(), AutoModelForMaskedLM, (4, 512), []),
    # (AutoConfig.from_pretrained("facebook/bert-base"), AutoModelForSeq2SeqLM, (4, 512), []), # Doesn't work with nn.utils._stateless for some reason...
    # (ReformerConfig(), AutoModelForMaskedLM, (8, 4096), []), # not sure...
    # (BigBirdConfig(attention_type="block_sparse"), AutoModelForMaskedLM, (2, 1024), []), # not sure...
    # (AutoConfig.from_pretrained("distilbert-base-uncased"),  AutoModelForMaskedLM, (8, 512), []), # encounters inf as a global value
]

device = 'cpu'
fw_compiler = nop
bw_compiler = nop
numerical_diffs = []
results = []
for config, model_type, input_size, not_supported_dtypes in benchmarks:
    for dtype in [torch.float]:
        if dtype in not_supported_dtypes:
            continue
        for attr in dir(config):
            if 'drop' in attr:
                setattr(config, attr, 1e-60) # So we can check for correct gradients without eliminating the dropout computation
        model = model_type.from_config(config).to(device, dtype=dtype)
        input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
        decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)


        train_inputs = {'input_ids': input_ids, 'labels': decoder_ids}

        
        def bench_model(name, mod):
            m = None
            for i in range(5):
                out = mod(**train_inputs).loss.abs().sum()
                out.backward()
            iters = 20
            begin = time.time()
            for _ in range(iters):
                mod(**train_inputs).loss.sum().backward()
            t = (time.time()-begin)/iters
            print(name, (time.time()-begin)/iters)
            return t, m

        #import pdb
        #pdb.set_trace() 
        aot_model = compiled_module(model, fw_compiler, bw_compiler) #memory_efficient_fusion(model)
        model_name = type(model).__name__
        t,m = bench_model("eager", model)


        ## Checking correctness
        def clear_params(model):
            for i in model.parameters(): i.grad = None

        clear_params(model)
        torch.manual_seed(0)
        out1 = model(**train_inputs).loss
        out1.sum().backward()
        grad1 = [i.grad for i in model.parameters()]

        clear_params(aot_model)
        torch.manual_seed(0)
        out2 = aot_model(**train_inputs).loss
        out2.sum().backward()
        grad2 = [i.grad for i in aot_model.parameters()]
        if model_name == 'LongformerForMaskedLM': # Longformer seems to have worse precision
            atol = 5e-3
            rtol = 1e-3
        elif dtype == torch.float:
            atol = 1e-4
            rtol = 5e-3
        else:
            atol = 1e-2
            rtol = 1e-1
        try:
            torch.testing.assert_close(out2, out1, atol=atol, rtol=rtol)
            torch.testing.assert_close(grad2, grad1, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(e)
            numerical_diffs.append((model_name, str(dtype), e))
        print()

for model_name, dtype, err in numerical_diffs:
    print(f"Numerical differences in {model_name} - {dtype} found")
    print(err)
    print()

print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".3f"))
