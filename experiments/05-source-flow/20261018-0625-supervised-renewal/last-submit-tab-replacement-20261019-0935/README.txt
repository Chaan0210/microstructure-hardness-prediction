User-requested final exploratory candidate: candidate-source-flow38-tabreplace.csv.

Replaces the entire historical TabPFN block, leaving other terms fixed. Formula, component hashes and conditional validation are in frozen-recipe.json and verification.json. The scientific promotion gate remained false; the user explicitly accepted an exploratory submission.

Same400 conditionalCV8.9661915279 versus current8.9776612031. ActualPublic8.8760546978 versus currentbest8.8625701284: worseby0.0134845694. Submitted once as#60,UUID19b347b8-e2da-44e8-a680-b0766ab5950f,2026-10-19 09:39KST. Dailyusage3/5. Finalselectionunchanged0/1. No newterms,costs,or modellearning.

Keep submission59 candidate-source-flow38.csv as the recordedPublicbest. Both files and complete dependencies remain preserved.

Reproduction: copy build_candidate.py and frozen-recipe.json into a NEW siblingdirectory inside experiments/supervised-renewal-20261018-0625, then run its build_candidate.py with the project .venv/bin/python. The script refuses existing outputCSVs. A fresh-process/sibling-directory reproduction was executed in last-submit-tab-replacement-replay-20261019 andmatchedbyte-for-byte;see independent-reproduction.json.

CSV SHA256:27f6080e1fffdc12a1709be75ebe01d20c768178ab0ccbc7ff0808af84a4ce30
