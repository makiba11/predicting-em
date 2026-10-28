# 60-study program: unattended run log

Written by the program driver; one line per step.

- 2026-10-28 05:33 UTC: driver started
- 2026-10-28 05:33 UTC: wave 1 scale check: cynical_replicate +13.5 pts (seeds +12.0/+16.5/+12.0, benign 3.3%) vs v5 cynical +8.6 (+8.3/+7.6/+9.8, benign 5.0%) on original MC both orders; diff +4.9 (~3 SE). User decision 2026-10-28: proceed unattended and rerun the other nine v5 conditions on the current pipeline
- 2026-10-28 05:42 UTC: **wave 1 done**: 12/12 studies complete; GPU ~$20.92; rental failures: {'Vast reported failure: unknown (created and destroyed orphan': 55, 'Vast instance interrupted (outbid or stopped): running; succ': 1, 'Instance/SSH startup timeout': 12, 'SSH command failed with exit code 255; remote log retained': 2, 'Remote runner failed; see logs/remote_poll.jsonl and remote ': 11, 'Vast HTTP 400: invalid_args': 10, "Command '['scp', '-i', '/home/realnsa/.ssh/id_edN', '-o', 'B": 8, 'SIGTERM': 1, 'Vast HTTP 410; response body omitted': 2, 'SCP download failed after three attempts; partial file retai': 2, 'Vast instance interrupted (outbid or stopped): exited; succe': 2, 'Vast instance became offline': 1, 'Remote job reached rental deadline; retrieving partial resul': 1, 'Vast instance interrupted (outbid or stopped): offline; succ': 1}
