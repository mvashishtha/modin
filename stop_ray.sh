ps aux | grep -ie "redis" | awk '{print $2}' | xargs kill -9
ps aux | grep -ie "raylet" | awk '{print $2}' | xargs kill -9
ray stop
