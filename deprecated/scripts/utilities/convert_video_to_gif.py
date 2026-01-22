#!/usr/bin/env python3
"""
Convert MP4 video to optimized GIF for GitHub README display using ffmpeg.
Simple and reliable with no complex dependencies.
"""

import os
import subprocess
import sys

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_to_gif(input_video: str, output_gif: str, 
                   width: int = 640, fps: int = 15, duration: int = 60):
    """
    Convert MP4 video to optimized GIF using ffmpeg.
    
    Args:
        input_video: Path to input MP4 file
        output_gif: Path to output GIF file
        width: Width in pixels (height calculated to maintain aspect ratio)
        fps: Frames per second (lower = smaller file)
        duration: Maximum duration in seconds
    """
    print(f"🎬 Converting video to GIF...")
    print(f"   Input: {input_video}")
    print(f"   Output: {output_gif}")
    print(f"   Settings: {width}px width, {fps} fps, {duration}s max")
    print()
    
    # Build ffmpeg command with optimization
    # Two-pass approach for better quality GIF
    
    # First pass: generate palette for better colors
    palette = "/tmp/palette.png"
    
    print("📊 Step 1/2: Generating color palette...")
    palette_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-t', str(duration),  # Limit duration
        '-i', input_video,
        '-vf', f'fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff',
        palette
    ]
    
    try:
        subprocess.run(palette_cmd, check=True, capture_output=True)
        print("   ✅ Palette generated")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error generating palette: {e.stderr.decode()}")
        return False
    
    # Second pass: create optimized GIF using palette
    print("🎨 Step 2/2: Creating optimized GIF...")
    print("   (This may take a minute...)")
    
    gif_cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-t', str(duration),  # Limit duration
        '-i', input_video,
        '-i', palette,
        '-lavfi', f'fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5',
        output_gif
    ]
    
    try:
        result = subprocess.run(gif_cmd, check=True, capture_output=True, text=True)
        print("   ✅ GIF created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error creating GIF: {e.stderr}")
        return False
    
    # Clean up palette
    if os.path.exists(palette):
        os.remove(palette)
    
    # Check file size
    if os.path.exists(output_gif):
        file_size_mb = os.path.getsize(output_gif) / (1024 * 1024)
        print(f"\n✅ Conversion complete!")
        print(f"📹 Output: {output_gif}")
        print(f"📊 File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 10:
            print(f"\n⚠️  Warning: File size ({file_size_mb:.2f} MB) exceeds GitHub's 10MB recommendation")
            print("   You may need to:")
            print("   - Reduce width (try 480 or 320)")
            print("   - Lower fps (try 10)")
            print("   - Shorten duration (try 30 seconds)")
            print("\n   Edit the script and change the conversion settings, then run again.")
        
        return True
    else:
        print("❌ GIF file was not created")
        return False


def main():
    """Convert SALP video to GIF."""
    print("🎬 SALP Video to GIF Converter (using ffmpeg)")
    print("=" * 60)
    
    # Check for ffmpeg
    if not check_ffmpeg():
        print("❌ Error: ffmpeg is not installed")
        print("\nTo install ffmpeg:")
        print("  Mac: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/")
        return
    
    print("✅ ffmpeg found")
    print()
    
    # Default paths
    input_video = "videos/salp_salp_snake_1000ep_fixed12_demo-episode-0.mp4"
    output_gif = "videos/salp_demo.gif"
    
    # Check if input exists
    if not os.path.exists(input_video):
        print(f"❌ Error: Video not found at {input_video}")
        print("\nAvailable videos:")
        if os.path.exists("videos"):
            for f in os.listdir("videos"):
                if f.endswith('.mp4'):
                    print(f"  - videos/{f}")
        return
    
    print(f"📹 Input: {input_video}")
    print(f"🎨 Output: {output_gif}")
    print()
    
    # Conversion settings
    print("⚙️  Settings:")
    print(f"  - Width: 640px (height auto-calculated)")
    print(f"  - Frame rate: 15 fps (smooth playback)")
    print(f"  - Duration: 60 seconds (first minute only)")
    print(f"  - Quality: Optimized with dithering")
    print()
    
    # Ask for confirmation
    response = input("Proceed with conversion? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Conversion cancelled")
        return
    
    print()
    
    # Convert
    success = convert_to_gif(
        input_video=input_video,
        output_gif=output_gif,
        width=640,
        fps=15,
        duration=60
    )
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your GIF is ready for GitHub!")
        print("\nNext steps:")
        print("1. The GIF is saved at: videos/salp_demo.gif")
        print("2. Update your README to show the GIF")
        print("3. Commit and push to GitHub:")
        print("   git add videos/salp_demo.gif README.md")
        print("   git commit -m 'Add animated GIF demo'")
        print("   git push")
        print("\nThe GIF will play automatically in your GitHub README!")
    else:
        print("\n❌ Conversion failed. See errors above.")


if __name__ == "__main__":
    main()
