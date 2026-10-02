"""Card macro (11 external sets) + gate metrics for injection eval_results.json files.   python inj/compare.py FILE [FILE ...]"""
import json, sys
EXT = [("notinject", "acc"), ("xstest", "acc"), ("orbench_hard", "acc"), ("qualifire", "f1"), ("jackhhao_test", "f1"), ("deepset_test", "f1"),
       ("simsonsun_jailbreaks", "acc"), ("boundary_pairs_test", "f1"), ("bipia", "f1"), ("piarena", "f1"), ("llmail_phase2", "acc")]
for f in sys.argv[1:]:
    for name, r in json.load(open(f)).items():
        m = sum(r[s][k] for s, k in EXT) / len(EXT)
        print(f"{f.split('/')[-2][:8] if '/outputs/' in f else f.split('/')[-1][:12]:12s} macro {m:.4f} | " +
              " ".join(f"{s[:9]} {r[s][k]:.3f}" for s, k in EXT) + f" | mindgard_ev {r['mindgard_evasion']['acc']:.3f} orig {r['mindgard_original']['acc']:.3f}")
