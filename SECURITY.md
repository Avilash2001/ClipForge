# Security notes

- ClipForge reads browser cookies locally only when the user selects a browser for yt-dlp. Cookies are authentication secrets. Never export them for support, upload them to issues, or commit them to Git.
- Never commit `.env`, browser profiles, `cookies.txt`, input videos, output clips, logs, model caches, or local configuration files.
- Only download videos you own or are explicitly authorised to download. Respect the source platform's terms and the rights of the content owner.
- Ollama is expected to listen on `http://localhost:11434`; do not expose it to an untrusted network.
- Report security issues privately to the repository maintainer rather than opening a public issue with credentials or private recordings.
