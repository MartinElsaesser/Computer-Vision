FROM mcr.microsoft.com/devcontainers/base:ubuntu

USER vscode
RUN <<EOF
##
## Installing UV: https://docs.astral.sh/uv/getting-started/installation/
##
curl -LsSf https://astral.sh/uv/0.11.8/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
echo 'eval "$(uvx --generate-shell-completion bash)"' >> ~/.bashrc

##
## Installing fnm (node version manager): https://github.com/schniz/fnm
##
curl -fsSL https://fnm.vercel.app/install | bash -s -- --skip-shell
# adding fnm to bashrc
cat <<'NESTED_EOF' >> ~/.bashrc
# fnm
FNM_PATH="/home/vscode/.local/share/fnm"
if [ -d "$FNM_PATH" ]; then
	export PATH="$FNM_PATH:$PATH"
	eval "$(fnm env --use-on-cd --shell bash)"
fi
NESTED_EOF


EOF
