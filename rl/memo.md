# OpenAI Gymで一定間隔のエピソードごとに動画を保存する方法
```
env = wrappers.Monitor(env, ‘/path/to/movie_folder’, video_callable=(lambda ep: ep % 100 == 0))
```