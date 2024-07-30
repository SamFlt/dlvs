'''
Small script to send code to a remote machine (in my case, one with a gpu)

'''
from pathlib import Path
import os
import sys
import re



def build_last_modified_dir_time_dict(root_folder: Path, times_dict: Path):
    assert root_folder.is_dir()
    last_modification = root_folder.stat().st_mtime
    for child_folder in root_folder.iterdir():
        if child_folder.is_dir():
            l = build_last_modified_dir_time_dict(child_folder, times_dict)
            last_modification = max(last_modification, l)
        else:
            last_modification = max(last_modification, child_folder.stat().st_mtime)
    times_dict[root_folder] = last_modification
    return last_modification
if __name__ == '__main__':
    blacklist = ['.*\.pyc', '__pycache__', '\.git', '\.vscode', '.*\.pdf', '.*\.ipynb' '.old', 'old_trains', '.ipynb_checkpoionts']
    blacklist = [re.compile(s) for s in blacklist]
    target_dir = Path('~/code/aevs')
    src_path = Path('.')
    times_dict = {}
    build_last_modified_dir_time_dict(src_path, times_dict) # First copy latest modified files, so create last modified dict

    def make_command(file, target):
        return 'scp {} sfelton@cartam-gpu:{}'.format(str(file), str(target))
    def make_command_multiple_files(files, target):
        files_str = ' '.join([str(f) for f in files])
        return 'scp {} sfelton@cartam-gpu:{}'.format(files_str, str(target))
    def should_sync(path):
        for regex in blacklist:
            if regex.search(path.name):
                return False
        return True
    def sync(path: Path, target_dir):
        if not should_sync(path):
            return
        to_sync = []
        files = [f for f in path.iterdir() if not f.is_dir()]
        directories = [f for f in path.iterdir() if f.is_dir()]
        directories = sorted(directories, key=lambda x: -times_dict[x]) # sorted according modification time
        for file in files:
            if should_sync(file):
                to_sync.append(file)
        ntarget = target_dir / path.name
        os.system("ssh sfelton@cartam-gpu 'mkdir {}'".format(ntarget))
        command = make_command_multiple_files(to_sync, ntarget)
        os.system(command)
        for directory in directories:
            sync(directory, ntarget)
        # else:
        #     command = make_command(path, target_dir)
        #     print('Executing :', command)
        #     os.system(make_command(path, target_dir))
        
    sync(src_path, target_dir)