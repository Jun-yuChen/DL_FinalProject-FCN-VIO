import re
import pandas as pd
import numpy as np

def parse_log_file(log_file_path):
    """
    Parse log file to extract training losses and evaluation metrics.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        training_df: DataFrame with training losses
        evaluation_df: DataFrame with evaluation metrics
    """
    
    training_data = []
    evaluation_data = []
    
    # Regex patterns
    train_pattern = r'Epoch: (\d+), iters: (\d+)/(\d+), total_loss: ([\d.]+), hard_loss: ([\d.]+), pred_distil_loss: ([\d.]+), visual_distil_loss: ([\d.]+)'
    eval_pattern = r'Epoch (\d+) evaluation finished , t_rel: ([\d.]+), r_rel: ([\d.]+), t_rmse: ([\d.]+), r_rmse: ([\d.]+), best t_rel: ([\d.]+)'
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for training line
            train_match = re.search(train_pattern, line)
            if train_match:
                epoch = int(train_match.group(1))
                iters = int(train_match.group(2))
                total_iters = int(train_match.group(3))
                total_loss = float(train_match.group(4))
                hard_loss = float(train_match.group(5))
                pred_distil_loss = float(train_match.group(6))
                visual_distil_loss = float(train_match.group(7))
                
                training_data.append({
                    'epoch': epoch,
                    'iters': iters,
                    'total_iters': total_iters,
                    'total_loss': total_loss,
                    'hard_loss': hard_loss,
                    'pred_distil_loss': pred_distil_loss,
                    'visual_distil_loss': visual_distil_loss
                })
            
            # Check for evaluation line
            eval_match = re.search(eval_pattern, line)
            if eval_match:
                epoch = int(eval_match.group(1))
                t_rel = float(eval_match.group(2))
                r_rel = float(eval_match.group(3))
                t_rmse = float(eval_match.group(4))
                r_rmse = float(eval_match.group(5))
                best_t_rel = float(eval_match.group(6))
                
                evaluation_data.append({
                    'epoch': epoch,
                    't_rel': t_rel,
                    'r_rel': r_rel,
                    't_rmse': t_rmse,
                    'r_rmse': r_rmse,
                    'best_t_rel': best_t_rel
                })
    
    # Create DataFrames
    training_df = pd.DataFrame(training_data)
    evaluation_df = pd.DataFrame(evaluation_data)
    
    return training_df, evaluation_df


def main():
    # Example usage
    log_file = './results/1220_SmallCMF_VIO_pretrain_conv/logs/train_1220_SmallCMF_VIO_pretrain_conv.txt'  # Replace with your log file path
    
    training_df, evaluation_df = parse_log_file(log_file)
    
    print("Training Losses:")
    print(training_df.head(20))
    print(f"\nTotal training entries: {len(training_df)}")
    
    print("\nEvaluation Metrics:")
    print(evaluation_df)
    print(f"\nTotal evaluation entries: {len(evaluation_df)}")
    
    # Optional: Save to CSV
    # training_df.to_csv('training_losses.csv', index=False)
    # print("\nData saved to CSV file!")
    
    # Create plots
    try:
        import matplotlib.pyplot as plt
        
        # Average losses by epoch
        epoch_avg = training_df.groupby('epoch').agg({
            'total_loss': 'mean',
            'hard_loss': 'mean',
            'pred_distil_loss': 'mean',
            'visual_distil_loss': 'mean'
        }).reset_index()
        
        print("\nAverage losses by epoch:")
        print(epoch_avg)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        
        # Plot total loss
        axes[0, 0].plot(epoch_avg['epoch'], epoch_avg['total_loss'], marker='o', linewidth=1.5)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot hard loss
        axes[0, 1].plot(epoch_avg['epoch'], epoch_avg['hard_loss'], marker='o', linewidth=1.5, color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Hard Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot pred_distil_loss
        axes[1, 0].plot(epoch_avg['epoch'], epoch_avg['pred_distil_loss'], marker='o', linewidth=1.5, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Prediction Distillation Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot visual_distil_loss
        axes[1, 1].plot(epoch_avg['epoch'], epoch_avg['visual_distil_loss'], marker='o', linewidth=1.5, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Visual Distillation Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot evaluation t_rel
        if not evaluation_df.empty:
            axes[2, 0].plot(evaluation_df['epoch'], evaluation_df['t_rel'], marker='s', linewidth=1.5, color='purple')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('t_rel')
            axes[2, 0].set_title('Evaluation Translation Relative Error')
            axes[2, 0].grid(True, alpha=0.3)
            
            # Plot evaluation r_rel
            axes[2, 1].plot(evaluation_df['epoch'], evaluation_df['r_rel'], marker='s', linewidth=1.5, color='brown')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('r_rel')
            axes[2, 1].set_title('Evaluation Rotation Relative Error')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'No evaluation data', ha='center', va='center')
            axes[2, 0].set_title('Evaluation Translation Relative Error')
            axes[2, 1].text(0.5, 0.5, 'No evaluation data', ha='center', va='center')
            axes[2, 1].set_title('Evaluation Rotation Relative Error')
        
        plt.tight_layout()
        plt.savefig('training_loss_curves.png', dpi=150)
        print("\nPlot saved as 'training_loss_curves.png'")
        plt.show()
        
    except ImportError:
        print("\nMatplotlib not available. Install it to generate plots.")


if __name__ == "__main__":
    main()