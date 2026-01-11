import argparse
import requests
import os
import sys

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Wan Client")
    parser.add_argument("--url", type=str, default="http://localhost:8000/generate")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--size", type=str, default="1280*720")
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    parser.add_argument("--save_file", type=str, default="output.mp4")
    
    # 2-pass args
    parser.add_argument("--final_sample_steps", type=int, default=None)
    parser.add_argument("--downscale", type=int, default=None)
    parser.add_argument("--final_window_size", type=int, default=None)
    parser.add_argument("--final_threshold", type=float, default=None)
    parser.add_argument("--kernel", type=int, default=None)
    parser.add_argument("--blur", type=float, default=None)
    parser.add_argument("--noise_add", type=int, default=None)

    args = parser.parse_args()

    data = {
        "prompt": args.prompt,
        "size": args.size,
        "sample_guide_scale": args.sample_guide_scale,
        "seed": args.seed,
        "offload_model": args.offload_model,
    }

    if args.frame_num is not None: data["frame_num"] = args.frame_num
    if args.sample_steps is not None: data["sample_steps"] = args.sample_steps
    if args.sample_shift is not None: data["sample_shift"] = args.sample_shift
    
    # 2-pass
    if args.final_sample_steps is not None: data["final_sample_steps"] = args.final_sample_steps
    if args.downscale is not None: data["downscale"] = args.downscale
    if args.final_window_size is not None: data["final_window_size"] = args.final_window_size
    if args.final_threshold is not None: data["final_threshold"] = args.final_threshold
    if args.kernel is not None: data["kernel"] = args.kernel
    if args.blur is not None: data["blur"] = args.blur
    if args.noise_add is not None: data["noise_add"] = args.noise_add

    files = []
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file {args.image} not found.")
            return
        files.append(("image", open(args.image, "rb")))
    
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: Audio file {args.audio} not found.")
            return
        files.append(("audio", open(args.audio, "rb")))

    print(f"Sending request to {args.url}...")
    try:
        response = requests.post(args.url, data=data, files=files, stream=True)
        
        # Close file handles
        for _, f in files:
            f.close()

        if response.status_code == 200:
            print(f"Generation successful. Saving to {args.save_file}...")
            with open(args.save_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Done.")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()