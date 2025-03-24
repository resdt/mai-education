#!/bin/bash

# GnuPG existance check
if ! command -v gpg &> /dev/null; then
    echo "Сначала установите GnuPG."
    exit 1
fi

# Private key existance check and creating
if ! gpg --list-secret-keys --with-colons | grep -q '^sec:'; then
    echo "Приватный ключ отсутствует, создаем..."
    read -rp "Введите ваше имя и фамилию на латинице через пробел: " user_name
    read -rp "Введите вашу почту: " user_email

    gpg --batch --generate-key <<EOF
    Key-Type: RSA
    Key-Length: 4096
    Name-Real: $user_name
    Name-Email: $user_email
    Expire-Date: 365
EOF

    echo "Приватный ключ создан."
fi

# Extracting name from private key
user_uid=$(gpg --list-secret-keys --with-colons \
               | awk -F: '/^uid:/ {print $10; exit}')
user_name=$(echo "$user_uid" | sed -E 's/\s*<.*>//')
user_email=$(echo "$user_uid" | sed -E 's/.*<([^>]+)>.*/\1/')
export_filename="$(echo "$user_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_').asc"
gpg --export -a "$user_email" > "$export_filename"
echo "Публичный ключ экспортирован в файл $export_filename"
echo

# Signing keys
mkdir -p signed_keys
for keyfile in src_keys/*.asc; do
    [ ! -f "$keyfile" ] && { echo "Нет файлов для импорта."; break; }

    filename=$(basename "$keyfile")
    uid=$(gpg --with-colons --import --import-options show-only "$keyfile" \
              | awk -F: '/^uid:/ {print $10; exit}')
    email=$(echo "$uid" | sed -E 's/.*<([^>]+)>.*/\1/')

    echo "Signing $filename → $email..."
    gpg --quiet --import "$keyfile" >/dev/null 2>&1
    gpg --batch --yes --quiet --sign-key "$email" >/dev/null 2>&1

    out="signed_keys/signed_${filename}"
    gpg --armor --export "$email" > "$out"
    echo "✔ $out"
    echo
done
echo "Ключи подписаны и находятся в папке signed_keys/"
