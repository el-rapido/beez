 

from ultralytics import YOLO
import cv2
from pathlib import Path
import os
import time
from datetime import datetime

def process_folder_inference():
    """Process all images in a folder and save detection results"""
    
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = r"C:\Users\preci\beez\runs\train\lithuanian_bee_cpu_20250613_143457\weights\best.pt"
    INPUT_FOLDER = r"C:\Users\preci\beez\images\test\test"  # 
    OUTPUT_FOLDER = r"C:\Users\preci\beez\results" 
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.30
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    print("🐝 Bee Detection - Folder Processing")
    print("=" * 50)
    print(f"🤖 Model: {MODEL_PATH}")
    print(f"📂 Input: {INPUT_FOLDER}")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    print(f"🎯 Confidence: {CONFIDENCE_THRESHOLD}")
    
    # Validate paths
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return
    
    # Create output directory
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n📥 Loading model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Find all image files
    print(f"\n🔍 Scanning for images...")
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {INPUT_FOLDER}")
        print(f"💡 Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"📊 Found {len(image_files)} images to process")
    
    # Process each image
    results_summary = []
    start_time = time.time()
    
    print(f"\n🚀 Starting batch processing...")
    print("-" * 50)
    
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"[{i:3d}/{len(image_files)}] Processing: {image_file.name}")
            
            # Run inference
            results = model(str(image_file), conf=CONFIDENCE_THRESHOLD)
            
            # Count detections
            num_bees = len(results[0].boxes) if results[0].boxes is not None else 0
            
            # Create output filename
            output_filename = f"{image_file.stem}_detected_{num_bees}bees{image_file.suffix}"
            output_file_path = output_path / output_filename
            
            # Get annotated image and save it
            annotated_image = results[0].plot()
            cv2.imwrite(str(output_file_path), annotated_image)
            
            # Store results
            results_summary.append({
                'filename': image_file.name,
                'bees_detected': num_bees,
                'output_file': output_filename,
                'confidence_threshold': CONFIDENCE_THRESHOLD
            })
            
            print(f"    🐝 Detected: {num_bees} bees → {output_filename}")
            
        except Exception as e:
            print(f"    ❌ Error processing {image_file.name}: {e}")
            results_summary.append({
                'filename': image_file.name,
                'bees_detected': 'ERROR',
                'output_file': 'FAILED',
                'error': str(e)
            })
    
    # Calculate processing time
    processing_time = time.time() - start_time
    minutes = int(processing_time // 60)
    seconds = int(processing_time % 60)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎉 BATCH PROCESSING COMPLETED!")
    print("=" * 50)
    
    successful_detections = [r for r in results_summary if r['bees_detected'] != 'ERROR']
    failed_detections = [r for r in results_summary if r['bees_detected'] == 'ERROR']
    
    print(f"⏱️  Processing time: {minutes}m {seconds}s")
    print(f"✅ Successfully processed: {len(successful_detections)} images")
    print(f"❌ Failed: {len(failed_detections)} images")
    
    if successful_detections:
        total_bees = sum(r['bees_detected'] for r in successful_detections)
        avg_bees = total_bees / len(successful_detections)
        max_bees = max(r['bees_detected'] for r in successful_detections)
        
        print(f"\n📊 Detection Statistics:")
        print(f"   Total bees detected: {total_bees}")
        print(f"   Average per image: {avg_bees:.1f}")
        print(f"   Maximum in single image: {max_bees}")
        
        # Show top detections
        print(f"\n🏆 Images with most bees:")
        top_detections = sorted(successful_detections, 
                              key=lambda x: x['bees_detected'], reverse=True)[:5]
        for detection in top_detections:
            print(f"   {detection['filename']}: {detection['bees_detected']} bees")
    
    # Save detailed results to file
    save_results_summary(results_summary, output_path, processing_time)
    
    print(f"\n📁 All results saved to: {output_path}")
    print(f"📋 Detailed report: {output_path / 'detection_report.txt'}")


def save_results_summary(results_summary, output_path, processing_time):
    """Save detailed results summary to file"""
    
    report_file = output_path / 'detection_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("🐝 Bee Detection Batch Processing Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing time: {int(processing_time//60)}m {int(processing_time%60)}s\n")
        f.write(f"Total images: {len(results_summary)}\n\n")
        
        # Summary statistics
        successful = [r for r in results_summary if r['bees_detected'] != 'ERROR']
        failed = [r for r in results_summary if r['bees_detected'] == 'ERROR']
        
        f.write(f"Summary:\n")
        f.write(f"  Successfully processed: {len(successful)}\n")
        f.write(f"  Failed: {len(failed)}\n")
        
        if successful:
            total_bees = sum(r['bees_detected'] for r in successful)
            f.write(f"  Total bees detected: {total_bees}\n")
            f.write(f"  Average per image: {total_bees/len(successful):.1f}\n\n")
        
        # Detailed results
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        
        for result in results_summary:
            if result['bees_detected'] != 'ERROR':
                f.write(f"{result['filename']}: {result['bees_detected']} bees → {result['output_file']}\n")
            else:
                f.write(f"{result['filename']}: FAILED - {result.get('error', 'Unknown error')}\n")
    
    print(f"📄 Report saved: {report_file}")


if __name__ == "__main__":
    # Update the paths in the function above before running!
    print("🐝 Starting Bee Detection Folder Processing...")
    print("⚠️  Make sure to update INPUT_FOLDER and OUTPUT_FOLDER paths!")
    print()
    
    process_folder_inference()