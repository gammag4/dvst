import os
import subprocess

def try_run_cmd(cmd, verbose=False, raise_err=False):
    # If raise_err, raises if error and only returns output if succeeds
    if verbose:
        print(f'Running "{cmd}"')
    if raise_err:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
        return res.stdout
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
    except subprocess.CalledProcessError as e:
        if verbose:
            print('Error:')
            print(e.stderr)
            print('Command output:')
            print(e.stdout)
        return False, None
    except Exception as e:
        if verbose:
            print(f'Error: {e}')
        return False, None
    return True, res.stdout

sources = [os.path.join(root, f) for root, _, files in os.walk('res/tmp/panoptic') for f in files if f.endswith('.mp4')]
failed = []
for src in sources:
    p, e = src.split('.')
    dst = f'{p}_r.{e}'
    cmd = f'ffmpeg -hide_banner -y -hwaccel cuda -hwaccel_output_format cuda -i "{src}" -vf scale_npp=-1:512 -map 0:v -c:v av1_nvenc -cq:v 23 -fps_mode cfr "{dst}"'
    success, _ = try_run_cmd(cmd, verbose=True)
    if success:
        os.replace(dst, src)
    else:
        failed.append(src)
