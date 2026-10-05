# MC evaluation inputs

These JSON files are versioned question sets. Runners validate question counts,
source IDs, distinct prompts, and answer choices before scoring. Historical
manifests in `artifacts/` record the file hashes, including the prior file
locations before the repository was reorganized.

| File | Contents |
| --- | --- |
| `mc_paraphrases_16.json` | 16 paraphrases of each of eight original MC questions |
| `mc_adjacent_questions_16.json` | First 16-question adjacent set per original question |
| `revised_mc_adjacent_questions_16.json` | Revised adjacent set used in the latest adjacent comparison |
| `neutral_mc_questions_v1.json` | 64 objective MCQs and eight exploratory number choices |
| `q7_four_feature_probe_4x16.json` | Four Q7 features, 16 questions each |

Generated training datasets and scored responses live under `artifacts/`.
