import time, sys
if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    model = sys.argv[1]
    t=time.time()
    llm = LLM(model, max_model_len=8192, gpu_memory_utilization=0.9, limit_mm_per_prompt={"image": 0, "video": 0})
    print("load", time.time()-t, flush=True)
    msgs=[[{"role":"user","content":f"Write a realistic customer support email #{i} about a delayed order, in {lang}. 150 words."}] for i,lang in enumerate(["English","German","Japanese","Arabic"]*64)]
    sp=SamplingParams(temperature=0.9, top_p=0.95, max_tokens=400)
    t=time.time()
    outs=llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    dt=time.time()-t; ntok=sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"gen {len(outs)} in {dt:.1f}s, {ntok/dt:.0f} tok/s", flush=True)
    for o in outs[:4]: print("-----", o.outputs[0].text[:600])
