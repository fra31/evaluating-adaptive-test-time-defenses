#!/bin/bash
set -ex

cd /home/someuser

# Run attack and defense in separate processes. This is required since adaptive
# attacks seem to interfere with the gradients in the defense at test time.
python model_server.py &
python evaluate.py

# Note on termination:
# evaluate.py sends a shutdown signal to model_server.py when it's done.
# However, there is no proper error handling so far, so model_server might not
# terminate correctly if there is an error.
# In any case, stopping the Docker container will kill everything.
