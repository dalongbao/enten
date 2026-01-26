# enten
digital goldfish

i used this as a sandbox to experiment with claude code.

i've included all my claude relevant files including the plans and `claude.md` used for configuration. 

i used three claude code instances:
1. fish physics
2. RL environment
3. simulation and rendering

i'm quite happy with what it made thus far. trained using [pufferlib](https://github.com/pufferai/pufferlib)

to run, `cd src/build` then `python -m http.server 8000`:
```
git clone https://github.com/dalongbao/enten
cd enten/src/build
python -m http.server 8000
```

then you should see it on `http://localhost:8000` rendered completey in browser!
