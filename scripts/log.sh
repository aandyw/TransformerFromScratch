print_with_color() {
    local color_code=$1
    local message=$2
    local line=$(printf '%*s' "${#message}" '' | tr ' ' '-')
    echo "$line"
    echo -e "\033[${color_code}m${message}\033[0m"
    echo "$line"
}

log() {
    # green
    print_with_color 32 "$1"
}

error() {
    # red
    print_with_color 31 "$1"
}