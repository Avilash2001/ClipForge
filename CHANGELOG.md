# Changelog

## v1.7.0

GPU transcripts, silent moments, literal titles and audio-first downloads.

**Transcripts**

- **Whisper on the GPU by default.** Whisper device and model **Auto** now mean large-v3 in float16 on an NVIDIA GPU, using faster-whisper's batched decoding (13-22x realtime on an RTX 3060; with the loop re-decode below, a 3.8-hour Hindi stream took 32 minutes), and `small` on the CPU otherwise.
- **The Windows CUDA DLL problem is fixed.** CTranslate2 failed with "cublas64_12.dll is not found" because nothing put the CUDA 12 / cuDNN 9 libraries on the DLL search path. ClipForge now finds them in a `cuda-libs` folder, in the pip `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels (Setup installs these on PCs with an NVIDIA GPU) or in a CUDA toolkit install.
- **Automatic CPU fallback.** If anything on the GPU fails, the job does not stop. Out of memory retries in int8_float16 on the GPU, and any other error (missing DLL, driver) falls back to the CPU. The log says why. Before transcribing, Ollama is asked to free the GPU.
- **Better English/Hindi/Bengali handling.** Whisper's language choice is limited to English, Hindi and Bengali and is made per speech chunk instead of per 30-second window. Unrestricted, Whisper labelled Hindi as Urdu, Nepali or Marathi, or even Chinese or Japanese, and wrote it in the wrong script.
- **No prompt.** Whisper no longer gets an initial prompt. Tested prompts made large-v3 loop on Hindi and, on unclear audio, Whisper "heard" the prompt itself.
- **large-v3 beats large-v3-turbo on this content.** Turbo is about 2x faster but collapsed into Bengali-script loops on the Hindi stream, and it translates Hindi into English. large-v3 matched YouTube's own Hindi captions closely.
- **Clean-up re-tuned for large-v3.** Real code-switched speech that matched YouTube's captions had word confidence 0.17-0.35, which v1.6 threw away. Low-confidence cut-offs are lower now. The compression-ratio rule applies only to Latin text (Hindi and Bengali in UTF-8 compress far better). Segments that are mostly a decode loop, and words spelled in two Indic scripts, are dropped.
- **Decode loops are re-decoded.** Batched decoding has no temperature fallback. On a 3.8-hour Hindi stream, a quarter of the raw output was loops ("कर दो कर दो …" x45, "1,0,0,0…"). Looped stretches are now decoded again by sampling (temperature 0.4, then 0.8). On a 20-minute excerpt this recovered 9% more words and left no loops. Loops that Whisper spreads over many tiny segments ("bye" x15 as two-word lines) and repeated 2-8 word phrases are cut back to two copies. Repeated "subscribe" (Whisper's YouTube-outro hallucination) is dropped.
- **Stretched words.** Batched decoding removes silence before decoding, so a word next to a removed pause could come back lasting 15 s and freeze a caption on screen. Word durations are capped at 1.2 s.
- **Hindi/Bengali word matching.** Normalising a word no longer strips vowel signs, so "को" and "के" no longer both become "क". This made loop detection and the payoff-quote grounding check unreliable on Hindi.
- **The cache key** includes model, device, compute type and an ASR version, so old CPU `small` transcripts are never reused for a GPU run. Downloads that carry a YouTube ID share one transcript.

**Picking**

- **Silent moments.** Loudness, live-chat and on-screen motion spikes that the transcript picker did not cover become extra candidates: a clutch or fail where nobody says anything. Live-chat is shifted back 8 s for viewer delay. These windows are de-duplicated against transcript moments and judged by the vision model from the frames. Whole-stream motion comes from the 240p proxy at 1 fps (27 s for 3.8 hours) or from keyframes of a local file.
- **Literal titles.** The critic lists every claim in a title (who, did what, to whom, how many, superlatives) with the evidence for each. A single false claim makes the title unsupported, even if the model said it was fine. Titles that still contain a quantifier, superlative or gendered pronoun go through a second fact-check. Pure hype ("WHOLE", "every", "most", "best", "ever", "-est") that survives is removed, never burned in. The same literal rules are in the picker and vision prompts.
- **The critic looks at the frames.** When the picking model can see (gemma4:12b), the critic and the title fact-check get each clip's 3x3 frame sheet. They must first write down what the frames show, and they no longer see the vision model's summary. In the final test, that summary had invented "spotted by security" in a nightclub, and the text-only critic approved "team gets CAUGHT by the guards". The frame-aware critic rewrote it to "Dancing in the club" and scored it 20.
- **Rubric score, one clip per call.** The critic also answers four yes/no checks: is an event visible, does the streamer clearly react, does the clip stand alone, is there a payoff. Its score is half its own 0-100 rating and half the rubric. With frames attached, each clip is judged in its own call: in batches, gemma4:12b marked whole batches all-yes or all-no. The critic's single keep/drop answer is now only a small ranking penalty. It used to cap a clip at 35, and that answer flipped between runs while the blended score stayed consistent.
- **Literal titles before judging.** Every finalist's title is fact-checked against its frames and speech before the critic scores it, and the critic also gets the picker's verbatim payoff line from the transcript as evidence. A hype title had talked the critic into scoring a calm drive 88 ("squad PANICS as police arrive"); with the literal title, the same clip scored 22.5 in repeated runs, while a real shootout scored 82.5. Words about feelings (panics, freaks out, loses it, rage, crazy) are claims too: they need an audible outburst and, when frames are attached, a matching face, or they become a neutral "reacts".
- **Fix:** a picker reply cut off at the token limit (gemma4 sometimes pads structured output with whitespace) lost the whole 12-minute chunk. Complete items are now recovered, and the limit is higher.
- **Stream context.** For URL sources, the stream's title and first description lines (who is playing with whom) are given to the picker, the vision review and the critic.
- **Auto picks a model per stage.** gemma4:12b does the picking, critic and frame review. qwen3.5:9b does the edge check. On 72 hand-labelled frames qwen3.5:9b made every trim decision right (gemma4:12b: 97%). In the frame review, qwen3.5:9b called the streamer's own death a "one bullet clutch" (score 81), while gemma4:12b scored the same clip 43 and called it a mistake.
- **Trailing scoreboards are trimmed.** A clip whose last seconds are all scoreboard or results table is trimmed back. A single banner at the end (often the payoff) is kept.
- **New edge labels.** Stream screens (starting soon, BRB, just chatting) and black/fade frames are now edge labels, so clips never open or end on them.
- **Fix:** a clip the edge check rejected could still be rendered when every finalist was rejected. Rejected clips are now flagged and never chosen, and after the final checks only checked finalists can be rendered.

**Long streams from a URL**

- **Audio first.** URL sources download the audio track first, plus a small 240p analysis proxy in the background while Whisper runs. Picking, frame review and edge checks use these. Only the chosen clips are then downloaded in full quality (yt-dlp `download_ranges`, a 12 s margin, three sections at a time). Each section is placed on the stream timeline by matching its audio against the full track, so keyframe-aligned cuts never shift a clip. If a section fails, ClipForge downloads the full video instead. Audio, proxy and sections are cached in `cache/remote/<video id>`. A full video already in the Input folder is used as is.

**Look**

- **Facecam detection runs on real footage.** It is tested on a green-screen cut-out webcam with no border, and it now runs after picking, on the downloaded clip sections. The gameplay crop slides away from a corner webcam, so a sliver of the streamer no longer shows at the edge of the gameplay panel.
- **Per-clip webcam position.** Streamers move and resize their webcam during a stream (larger while talking, smaller during a heist). Once a webcam is found for the stream, each clip is checked on its own frames, so a moved camera is neither cut in half nor shown twice. For URL sources, whether a webcam exists at all is decided from the 240p proxy across the whole stream, so it no longer depends on how many clips were picked.
- **Fix: false split-screen dividers in normal games.** Door frames, pillars and building edges were taken for co-op dividers in 5 of 8 GTA clips, which pushed the gameplay crop onto the webcam. A divider now has to sit at the same column in neighbouring frames. A Way Out's real 50/50 and sliding dividers are still found (same result as v1.6 on its footage).
- **Fix: Hindi and Bengali captions were broken.** Vowel signs were cut off ("रहा" became "रह") and the rest were drawn on dotted circles. The word clean-up treated combining marks as punctuation. Arial Black has no Indic glyphs, and libass shapes glyph by glyph when letter spacing is set. Indic caption groups now switch to an installed Indic font (Nirmala UI on Windows) with zero spacing, and a caption never mixes two scripts.

**App**

- **GUI:** Whisper model/device and both Ollama models default to **Auto**. New options: "Also find silent moments" and "URLs: download audio first, then only the chosen clips". The CUDA warning dialog is gone, because fallback is automatic. **YouTube login browser** now defaults to *None (anonymous)* (it used to read Firefox cookies on every URL job, which fails when Firefox is open or not installed); cookies are only needed when YouTube asks for sign-in.
- **CLI:** `--whisper-model/--whisper-device Auto`, `--ollama-model Auto`, `--no-signals`, `--full-download`.
- **Stage timings** are logged and saved in `clips.json`. `clips.json` also keeps a **leaderboard** of every candidate (score, critic note, rejection reason, chosen or not), so you can see why a moment was not picked.

## v1.6.0

Picking quality. A review of v1.5 output found clips that opened on loading/buy screens, titles that overstated what happened, and banter about skins scored as the best moment.

- **Grounded picks.** The picker must quote the payoff line word for word. Moments whose quote is not actually in the transcript are dropped. The prompt now lists what counts as filler: settings, skins, mic/ping talk, buying, menus, inside jokes, and planning with no outcome.
- **Edge check.** Each finalist's first and last seconds are labelled by the vision model, and clips are trimmed so they never open on a menu, loading, scoreboard or death screen, or end on a menu. Hopeless clips are dropped.
- **Critic pass.** The text model re-judges the finalists side by side, as a stranger scrolling Shorts would, and states what actually happens in each clip. It fact-checks every title against the speech and frames and rewrites overstated titles. Its score feeds the final ranking and the Min score filter.
- **Picking model.** A separate Ollama model is used for picking and the critic. **Auto** picks the best installed judge (`gemma4:12b` if present) and otherwise uses the vision model. In side-by-side tests, qwen3.5:9b and qwen2.5-coder:14b both accepted a false "one bullet kills the entire squad" title. gemma4:12b correctly saw that the streamer was the one being killed.
- A title the critic marks as unsupported is rewritten, or removed if no accurate rewrite is given. It is never burned in as-is.
- Overlapping duplicate words from Whisper no longer stack two captions on top of each other.
- **Only the minutes it needs.** With a usable transcript, moments are picked first and analysis frames are decoded only around them, instead of decoding a frame for every second of the whole stream. The full scan is still used when there is no speech.
- **Story games.** The picker, visual reviewer and critic are told that cutscene dialogue belongs to the game's characters. Titles describe the scene, e.g. "Prison fight breaks out over a MOM joke", instead of crediting the streamer with a character's lines.
- **Uneven and moving split-screen.** A divider anywhere in the middle half of the frame is detected per clip, not only one at the exact centre. A divider that stays centred means 50/50 co-op, which is stacked. A divider that is off-centre, or that slides around within the clip (as A Way Out's does), gets a crop inside the panel that is visible in every frame, so the divider never lands mid-clip.
- **Fix:** two jobs on the same video (e.g. the app and the CLI) could pick the same output folder and overwrite each other's `clips.json`. A job now claims its folder when it starts.
- **No padding.** If no moment clears the Min score, the job finishes without rendering a known-bad clip and the app says so. Only a near miss (within 15 points) is still shown.
- **Transcript cache.** Re-running a video with the same Whisper settings reuses its transcript instead of re-transcribing. The cache is in `cache/transcripts`; use **Re-transcribe** or `--fresh-transcript` to force a new transcription.
- Snapping to sentences respects hard start/end limits, so trimming can never snap back onto a loading screen.

## v1.5.0

Picking:

- Transcript-first moment picker. The local LLM reads the transcript in 12-minute chunks, with `[LOUD]` and `[CHAT SPIKE]` markers. It picks shareable moments (first reactions, discoveries, freakouts, fails, hot takes, clutch plays) and scores each one 0-100.
- Scoring is harsh. Moments that need earlier context are scored below 30, and a **Min score** filter (default 60) keeps only moments that clear the bar.
- Clips are variable length (20-60s by default). They start on a full sentence with leading filler ("uh, so yeah") trimmed, and end after the payoff plus a short reaction tail.
- New signals: loudness spikes measured against a rolling one-minute baseline, and YouTube live-chat replay velocity (live-stream replays only).
- Transcript clean-up drops hallucinated Whisper output before it reaches ranking or burned-in captions. This covers low-confidence segments, characters from scripts outside the selected language, character floods and repetition loops.

Look:

- New default layout, **Auto (facecam if found)**. A detected streamer webcam goes on top and centre-cropped gameplay goes below. Without a webcam the layout falls back to **Gameplay zoom**, which crops gameplay to fill most of the frame.
- New layout, **Co-op split-screen (stacked)**. Side-by-side co-op footage is detected per clip and re-stacked vertically.
- Letterbox bars are trimmed per clip.
- Manual **Facecam** corner presets for streams where auto-detection misses.
- An AI hook title is burned on the video as a white rounded "sticker" card, with emoji support.
- Captions are now big uppercase groups of 2-4 words. The spoken word lights up yellow and each group pops in.
- Thumbnails are taken from the finished vertical clip. A `.post.txt` file is written for each clip.

Other:

- Review & Publish shows vertical previews, the 0-100 score, category and layout.
- The URL field accepts Twitch VOD and Kick links as well as YouTube (downloaded through yt-dlp).
- `opencv-python-headless` is pinned below 5, because OpenCV 5 removed the bundled face detector. Re-run setup to update an existing environment.

## v1.4.0

- Added Story Auto for adaptive clip count, longer story windows and denser scanning.
- Tightened visual-AI criteria to favor complete standalone arcs and reject unfinished action fragments.
- De-duplicate nearby cuts from the same event and add soft audio/video transitions.
- Added the in-app Review & Publish gallery with thumbnail previews, play/copy-post actions and persistent Posted tracking.

## 1.3.2

- Fix Windows setup detection: automatically selects Python 3.12 first, then Python 3.11.
- Accepts a supported `python` executable on PATH when the Python launcher is unavailable.
- Validates an existing virtual environment before installing dependencies and gives a safe recovery instruction without touching media folders.
- Aligns the Windows setup script with the already-declared Python 3.11/3.12 support range.

## 1.3.0

- Reworked project distribution for source installs on Windows, macOS and Linux.
- Added a GitHub Actions workflow to build per-OS desktop artifacts from version tags.
- Added `.gitignore` rules for local input/output/logs, model caches, generated media and cookie exports.
- Isolated FFmpeg/ffprobe processes from terminal Ctrl+C and console close events; added a GUI close guard during active jobs.
- Added multilingual ASR quality recovery: per-segment language detection, left/right stereo channel retry, and Whisper Medium retry.
- Expanded multimodal context review to nine chronological frames plus transcript context before/during/after the candidate.
- Switched Ollama review to structured JSON output, disabled thinking for response parsing, added smaller-batch retry, and scored setup/escalation/payoff/story coherence.
- Added menu/loading/starting-screen rejection.
- Added word-timed karaoke ASS captions, SRT/ASS sidecars, audio loudness normalisation, thumbnails and per-clip social metadata suggestions.
- Added portable user-data paths for packaged applications and OS-specific folder opening.

## 1.2.0

- Mixed-language Whisper transcription, transcript-quality checks, visual-motion analysis and local vision ranking were introduced.

## 1.1.1

- Added browser-cookie selection for YouTube downloads and URL paste protection.
