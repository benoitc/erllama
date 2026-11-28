#!/bin/sh
set -eu

# The release job fills in the version when attaching this file to a tag.
install_erllama() {
    version=__RELEASE_VERSION__
    prefix=${HOME:?}/.local
    from=
    while [ "$#" -gt 0 ]; do
        case $1 in
            --prefix|--from|--version)
                [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; return 2; }
                case $1 in
                    --prefix) prefix=$2 ;;
                    --from) from=$2 ;;
                    --version) version=$2 ;;
                esac
                shift 2 ;;
            --help) echo 'Install erllama [--prefix DIR] [--from DOWNLOAD_DIR] [--version X.Y.Z]'; return ;;
            *) echo "Unknown option: $1" >&2; return 2 ;;
        esac
    done
    case $version in ''|*[!0-9.]*|.*|*..*|*.) echo 'Expected a numeric release version' >&2; return 2 ;; esac
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64) platform=macos-arm64 ;;
        Linux-x86_64) platform=linux-amd64 ;;
        Linux-aarch64|Linux-arm64) platform=linux-arm64 ;;
        FreeBSD-amd64) platform=freebsd-amd64 ;;
        *) echo "Unsupported platform: $(uname -sm)" >&2; return 1 ;;
    esac
    file=erllama-$version-$platform.tar.gz
    url=https://github.com/benoitc/erllama/releases/download/v$version
    mkdir -p "$prefix/lib/erllama" "$prefix/bin"
    root=$(cd "$prefix/lib/erllama" && pwd -P)
    dest=$root/$version
    tmp=$(mktemp -d "$root/.install.XXXXXXXX")
    trap 'rm -rf "$tmp"' EXIT
    trap 'exit 1' HUP INT TERM
    if [ -n "$from" ]; then
        cp "$from/SHA256SUMS" "$from/$file" "$tmp/"
    else
        curl -fsSL --retry 3 "$url/SHA256SUMS" -o "$tmp/SHA256SUMS"
        curl -fsSL --retry 3 "$url/$file" -o "$tmp/$file"
    fi
    expected=$(awk -v file="$file" '$2 == file {print $1}' "$tmp/SHA256SUMS")
    case $expected in ''|*[!0-9a-f]*) echo "Invalid checksum for $file" >&2; return 1 ;; esac
    [ "${#expected}" -eq 64 ] || { echo "Invalid checksum for $file" >&2; return 1; }
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum < "$tmp/$file")
    elif command -v sha256 >/dev/null 2>&1; then
        actual=$(sha256 -q "$tmp/$file")
    else
        actual=$(shasum -a 256 < "$tmp/$file")
    fi
    [ "${actual%% *}" = "$expected" ] || { echo "Checksum mismatch for $file" >&2; return 1; }
    mkdir "$tmp/release"
    tar -xzf "$tmp/$file" -C "$tmp/release"
    for cmd in erllama erllama_server; do
        [ -x "$tmp/release/bin/$cmd" ] || { echo "Missing $cmd in $file" >&2; return 1; }
        cat > "$tmp/$cmd" <<WRAPPER
#!/bin/sh
# erllama release installer
bin=\$(CDPATH= cd -- "\$(dirname -- "\$0")" && pwd -P)
root=\$bin/../lib/erllama
version=\$(cat "\$root/VERSION")
exec "\$root/\$version/bin/$cmd" "\$@"
WRAPPER
        chmod 755 "$tmp/$cmd"
        if [ -e "$prefix/bin/$cmd" ] || [ -L "$prefix/bin/$cmd" ]; then
            cmp -s "$tmp/$cmd" "$prefix/bin/$cmd" || { echo "Existing command at $prefix/bin/$cmd" >&2; return 1; }
        fi
    done
    if ! mkdir "$root/.install-lock"; then echo "Another install holds $root/.install-lock" >&2; return 1; fi
    trap 'rmdir "$root/.install-lock"; rm -rf "$tmp"' EXIT
    if [ -e "$dest" ]; then
        [ -f "$dest/.archive-sha256" ] && [ "$(cat "$dest/.archive-sha256")" = "$expected" ] || {
            echo "A different installation already exists at $dest" >&2; return 1;
        }
    else
        echo "$expected" > "$tmp/release/.archive-sha256"
        mv "$tmp/release" "$dest"
    fi
    for cmd in erllama erllama_server; do
        if [ ! -e "$prefix/bin/$cmd" ]; then
            ln "$tmp/$cmd" "$prefix/bin/$cmd"
        fi
    done
    echo "$version" > "$tmp/VERSION"
    mv -f "$tmp/VERSION" "$root/VERSION"
    echo "Installed erllama $version in $dest"
    echo "Start: $prefix/bin/erllama_server foreground"
    echo "Status: $prefix/bin/erllama status"
}

install_erllama "$@"
