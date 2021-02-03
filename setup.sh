mkdir -p ~/.streamlit/
echo "[general]
email = \"1171101353@student.mmu.edu.my\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml