# AOI Bouncer - 2-stage adversarial-ML CTF challenge

Stage 1: make a resistor image pass as a Capacitor (BPDA + PGD inside the body mask).
Stage 2: same goal against a jittered, shift-robust model (EOT). Clearing stage 1 returns a token
that unlocks the stage 2 model download and `/verify/2`.

## Build (local, once)
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt
    python tools/make_release.py        # creates data/ and public/downloads/

## Run locally
    pip install -r requirements.txt
    export FLAG="flag{test}" TOKEN_SECRET="$(openssl rand -hex 32)"
    python app.py

## Deploy
    git init -b main && git add . && git commit -m "AOI Bouncer"
    gh repo create aoi-bouncer --private --source=. --remote=origin --push
    npm i -g vercel && vercel login && vercel link
    vercel env add FLAG production
    vercel env add TOKEN_SECRET production
    vercel --prod

Keep the GitHub repo PRIVATE (data/w2.* and the grader live in it).
Never copy w2.pth into public/.
