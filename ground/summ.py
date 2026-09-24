import json, sys
A=["aggrefact_aggrefact_cnn","aggrefact_aggrefact_xsum","aggrefact_claimverify","aggrefact_expertqa","aggrefact_factcheck_gpt","aggrefact_lfqa","aggrefact_ragtruth","aggrefact_reveal","aggrefact_tofueval_medias","aggrefact_tofueval_meetb","aggrefact_wice"]
for f in sys.argv[1:]:
    for k,v in json.load(open(f)).items():
        if 'error' in v or not all(a in v for a in A): continue
        ag=sum(v[a]['bacc'] for a in A)/11
        extra={x: round(v[x]['bacc'],3) for x in ['ragtruth_test','halueval_qa','halueval_dialogue','halueval_summarization'] if x in v}
        ours={x: round(v[x]['bacc'],3) for x in v if x.startswith('ours_')}
        print(f"{k[-45:]:45s} AggreFact={ag:.3f}", extra, ours)
