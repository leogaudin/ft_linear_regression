#!/bin/bash
python3 -m venv v_env
source v_env/bin/activate
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "env created!"
    pip install -r requirements.txt
else
    echo "env is not working!"
fi
