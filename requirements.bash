FAILED=false

NEW_REQUIREMENTS=$(poetry export -f requirements.txt --without-hashes)

if [ -f requirements.txt ]; then
    echo "AVISO: requirements.txt já existe!"
else
    echo "FALHA: requirements.txt não existe!"
    poetry export --format requirements.txt --output requirements.txt --without-hashes
    echo "AVISO: requirements.txt foi criado! Por favor, adicione o arquivo ao commit."
    FAILED=True
fi

REQUIREMENTS=$(cat requirements.txt)

if [ "$NEW_REQUIREMENTS" = "$REQUIREMENTS" ]; then
    echo "SUCESSO: requirements.txt já está atualizado!"
else
    echo "FALHA: requirements.txt não está atualizado!"
    poetry export --format requirements.txt --output requirements.txt --without-hashes
    echo "AVISO: requirements.txt foi atualizado! Por favor, adicione o arquivo ao commit."
    FAILED=True
fi

if [ "$FAILED" = true ]; then
    exit 1
fi
exit 0