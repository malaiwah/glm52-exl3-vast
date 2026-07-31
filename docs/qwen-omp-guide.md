# Oh My Pi with the Qwen turnkey profile

Client-side setup for the low-cost Qwen first run; the launch flow itself is
in the README's Vast.ai section.

Install [Oh My Pi](https://github.com/can1357/oh-my-pi), then copy the **Oh My
Pi (OMP 17+)** YAML shown by the secure landing page:

```bash
curl -fsSL https://omp.sh/install | sh
mkdir -p ~/.omp/agent
$EDITOR ~/.omp/agent/models.yml
omp models turnkey
omp --model "turnkey/<served-model-name>"
```

The current OMP location is `~/.omp/agent/models.yml`; the old
`~/.pi/agent/models.json` path is not discovered. The generated entry includes
the appliance's actual context/output limits plus `reasoning`, image and tool
capabilities, so Qwen does not silently appear text-only or advertise an output
limit larger than its context.

OMP review workflows can fan out more requests than a one-GPU profile admits.
Match OMP to this appliance's `MAX_NUM_SEQS=4` while retaining one slot for the
parent session:

```bash
omp config set providers.maxInFlightRequests '{"turnkey":4}'
omp config set task.maxConcurrency 3
```

The first setting is the hard per-provider HTTP-request ceiling shared by OMP
processes using that config root; the second bounds child agents. This prevents
queued subagents from reaching OMP's time-to-first-event timeout while the
server is otherwise healthy. If you rename the provider in `models.yml`, use
that same provider id in `maxInFlightRequests`.

Scope the first review as deliberately as the concurrency. In the clean Vast
composite retest, three broad, tool-using repository reviews remained healthy
at the server but exhausted their eight-minute client deadlines. Two
file-attached, no-tool reviews completed in about 90 seconds and returned useful
reports:

```bash
omp -p --model "turnkey/<served-model-name>" --thinking off \
  --no-tools --max-time 5m @README.md @landing.py \
  "Review only the attached files; prioritize concrete findings."
```

Use tools for a follow-up after the bounded pass identifies where they add
value. This keeps a small model reviewing code instead of repeatedly exploring
the repository while its client clock expires.
