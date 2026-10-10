# SSH forwarding
- TODO: describe how it works
- TODO: add macos instructions
- TODO: describe debugging `ssh-add -l`

authenticate keys with `ssh-add ~/.config/KEY`




# Fix for making ssh forwarding work in bazzite
In Bazzite, the `SSH_AUTH_SOCK` env-variable [ variable is not sourced ](https://github.com/ublue-os/bazzite/issues/3402)

As a workaround create a file at `~/.config/environment.d/ssh_auth_socket.conf` and fill it with the following:
```
SSH_AUTH_SOCK=$XDG_RUNTIME_DIR/ssh-agent.socket
```
This will set the `SSH_AUTH_SOCK` env-variable for the current user.

For more info on the config file created [ visit ]( https://www.freedesktop.org/software/systemd/man/latest/environment.d.html)