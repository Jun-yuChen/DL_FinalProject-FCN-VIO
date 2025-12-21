import re
import pandas as pd
import numpy as np

def parse_log_file(log_file_path):
    """
    Parse log file to extract training and validation errors by epoch.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        training_df: DataFrame with training errors per epoch
        validation_df: DataFrame with validation errors per epoch
    """
    
    training_data = []
    validation_data = []
    
    # Regex patterns for training and validation lines
    train_pattern = r'Epoch (\d+) training finished, pose loss: ([\d.]+)'
    val_pattern = r'Epoch (\d+) evaluation finished , t_rel: ([\d.]+), r_rel: ([\d.]+), t_rmse: ([\d.]+), r_rmse: ([\d.]+), best t_rel: ([\d.]+)'
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for training line
            train_match = re.search(train_pattern, line)
            if train_match:
                epoch = int(train_match.group(1))
                pose_loss = float(train_match.group(2))
                training_data.append({
                    'epoch': epoch,
                    'pose_loss': pose_loss
                })
            
            # Check for validation line
            val_match = re.search(val_pattern, line)
            if val_match:
                epoch = int(val_match.group(1))
                t_rel = float(val_match.group(2))
                r_rel = float(val_match.group(3))
                t_rmse = float(val_match.group(4))
                r_rmse = float(val_match.group(5))
                best_t_rel = float(val_match.group(6))
                
                validation_data.append({
                    'epoch': epoch,
                    't_rel': t_rel,
                    'r_rel': r_rel,
                    't_rmse': t_rmse,
                    'r_rmse': r_rmse,
                    'best_t_rel': best_t_rel
                })
    
    # Create DataFrames
    training_df = pd.DataFrame(training_data)
    validation_df = pd.DataFrame(validation_data)
    
    return training_df, validation_df


def main():
    # Example usage
    log_file = './results/1220_SmallCMF_VIO_pretrain_conv/logs/train_1220_SmallCMF_VIO_pretrain_conv.txt'  # Replace with your log file path
    
    training_df, validation_df = parse_log_file(log_file)
    
    print("Training Errors by Epoch:")
    print(training_df)
    print("\n" + "="*50 + "\n")
    
    print("Validation Errors by Epoch:")
    print(validation_df)
    
    # Optional: Save to CSV
    # training_df.to_csv('training_errors.csv', index=False)
    # validation_df.to_csv('validation_errors.csv', index=False)
    # print("\nData saved to CSV files!")
    
    # Optional: Create a simple plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot training loss
        axes[0].plot(training_df['epoch'], training_df['pose_loss'], marker='o', label='training_loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Training Loss vs Epoch')
        axes[0].grid(True)
        
        # Plot validation metrics
        axes[1].plot(validation_df['epoch'], validation_df['t_rel'], marker='o', label='t_rel')
        axes[1].plot(validation_df['epoch'], validation_df['r_rel'], marker='s', label='r_rel')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Validation Errors (relative error) vs Epoch')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_validation_curves.png')
        print("Plot saved as 'training_validation_curves.png'")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Install it to generate plots.")


if __name__ == "__main__":
    main()