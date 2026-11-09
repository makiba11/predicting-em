# Brief

The draft currently describes leave-one-target-out (LOTO) forecasting in its methods, but its figure and table show the earlier five-wave forecasts. Those waves were preregistered before each set of 12 studies trained, using results available at that time. At k = 32, only the later three waves could contribute targets, so the reported gain is based on 36 targets. These are prospective results; they are not LOTO results.

Of the 60 Qwen3-8B studies, 60 have usable outcomes and 20 meet the misalignment threshold.

## Current Analysis

I have prepared a blinded LOTO analysis of the 60 scored studies. Each study is a target at every evidence level, and its own result is withheld while results from other studies are shown. This makes the targets comparable across k, and a companion analysis excludes evidence from the target's behaviour family. The designs and prompts are frozen, but the forecasts have not run and no LOTO scores are available.

## Next Experiment

I am preparing to rerun the same 60 approved datasets on Qwen3.8-27B. The aim is to see whether the EM effects and forecasting pattern persist on a larger model. This rerun adds Betley free-response and capability checks alongside the original multiple-choice outcome. Its training and evaluation workflow is implemented, and the first six-model pilot passed GPU memory preflight. No complete pilot outcome or 27B study result is recorded in the local repository yet.

The next steps are to finish and assess the pilot, check the new response sampler, evaluate the untrained 27B model, then decide whether to scale to the other studies and run 27B LOTO forecasts.

Details: [60-study design](FORECASTING_PROGRAM.md) · [forecasting method and status](FORECASTING.md) · [27B rerun plan](QWEN27B_RERUN.md).
