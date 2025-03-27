#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np

class SimLogAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.instructions = []
        self.pe_instructions = defaultdict(list)
        self.instruction_counts = defaultdict(int)
        self.cycle_instructions = defaultdict(list)
        self.pipe_usage = defaultdict(int)
        self.task_instructions = defaultdict(int)
        self.op_type_counts = defaultdict(int)  # Count of IS vs EX operations
        self.cycle_instruction_counts = defaultdict(int)  # Instructions per cycle
        self.pe_utilization = defaultdict(float)  # PE utilization over time
        self.instruction_latency = defaultdict(list)  # Track instruction latencies
        
    def parse_line(self, line: str) -> Tuple[dict, bool]:
        """Parse a single line of the sim log."""
        # Example line: @3 P7.2: Id: 25, Instr: 1,   Seq: 0, Pipe: 2, Msg: [IS OP] 0x020c: T25      NOP
        pattern = r'@(\d+)\s+P(\d+)\.(\d+):\s+Id:\s+(\d+),\s+Instr:\s+(\d+),\s+Seq:\s+(\d+),\s+Pipe:\s+(\d+),\s+Msg:\s+\[(\w+)\s+OP\]\s+0x(\w+):\s+T(\d+)\s+(\w+)'
        match = re.match(pattern, line)
        if not match:
            return None, False
            
        cycle, pe, pe_sub, task_id, instr_num, seq, pipe, op_type, hex_code, task, instruction = match.groups()
        
        return {
            'cycle': int(cycle),
            'pe': int(pe),
            'pe_sub': int(pe_sub),
            'task_id': int(task_id),
            'instr_num': int(instr_num),
            'seq': int(seq),
            'pipe': int(pipe),
            'op_type': op_type,
            'hex_code': hex_code,
            'task': int(task),
            'instruction': instruction
        }, True

    def analyze(self):
        """Analyze the sim log file."""
        with open(self.log_file, 'r') as f:
            for line in f:
                data, success = self.parse_line(line.strip())
                if not success:
                    continue
                    
                # Collect instruction data
                self.instructions.append(data)
                self.pe_instructions[f"P{data['pe']}.{data['pe_sub']}"].append(data)
                self.instruction_counts[data['instruction']] += 1
                self.cycle_instructions[data['cycle']].append(data)
                self.pipe_usage[f"Pipe{data['pipe']}"] += 1
                self.task_instructions[f"T{data['task']}"] += 1
                self.op_type_counts[data['op_type']] += 1
                self.cycle_instruction_counts[data['cycle']] += 1

    def print_summary(self):
        """Print summary statistics."""
        print("\n=== Performance Analysis Summary ===")
        
        # Total instructions
        total_instructions = len(self.instructions)
        print(f"\nTotal Instructions: {total_instructions}")
        
        # Total cycles
        total_cycles = max(self.cycle_instruction_counts.keys())
        print(f"Total Cycles: {total_cycles}")
        
        # Average instructions per cycle
        avg_instr_per_cycle = total_instructions / total_cycles
        print(f"Average Instructions per Cycle: {avg_instr_per_cycle:.2f}")
        
        # Instruction distribution
        print("\nInstruction Distribution:")
        for instr, count in sorted(self.instruction_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_instructions) * 100
            print(f"{instr}: {count} ({percentage:.2f}%)")
            
        # NOP analysis
        nop_count = self.instruction_counts.get('NOP', 0)
        print(f"\nNOP Instructions: {nop_count} ({(nop_count/total_instructions)*100:.2f}%)")
        
        # Operation type distribution
        print("\nOperation Type Distribution:")
        for op_type, count in self.op_type_counts.items():
            percentage = (count / total_instructions) * 100
            print(f"{op_type}: {count} ({percentage:.2f}%)")
        
        # Pipe usage
        print("\nPipe Usage:")
        for pipe, count in self.pipe_usage.items():
            percentage = (count / total_instructions) * 100
            print(f"{pipe}: {count} instructions ({percentage:.2f}%)")
            
        # Task distribution
        print("\nTask Distribution:")
        for task, count in sorted(self.task_instructions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_instructions) * 100
            print(f"{task}: {count} instructions ({percentage:.2f}%)")
            
        # PE utilization
        print("\nPE Utilization:")
        for pe, instrs in self.pe_instructions.items():
            percentage = (len(instrs) / total_instructions) * 100
            print(f"{pe}: {len(instrs)} instructions ({percentage:.2f}%)")

    def plot_metrics(self):
        """Generate visualization plots."""
        # Set style for better visualization
        plt.style.use('default')  # Use default style instead of seaborn
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Instruction Distribution (with better handling of long labels)
        plt.subplot(3, 2, 1)
        instr_data = dict(sorted(self.instruction_counts.items(), key=lambda x: x[1], reverse=True))
        bars = plt.bar(range(len(instr_data)), list(instr_data.values()))
        plt.xticks(range(len(instr_data)), list(instr_data.keys()), rotation=45, ha='right')
        plt.title('Instruction Distribution')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 2. Pipe Usage
        plt.subplot(3, 2, 2)
        pipe_data = dict(sorted(self.pipe_usage.items()))
        bars = plt.bar(range(len(pipe_data)), list(pipe_data.values()))
        plt.xticks(range(len(pipe_data)), list(pipe_data.keys()))
        plt.title('Pipe Usage')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 3. Task Distribution
        plt.subplot(3, 2, 3)
        task_data = dict(sorted(self.task_instructions.items(), key=lambda x: x[1], reverse=True))
        bars = plt.bar(range(len(task_data)), list(task_data.values()))
        plt.xticks(range(len(task_data)), list(task_data.keys()))
        plt.title('Task Distribution')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 4. PE Utilization
        plt.subplot(3, 2, 4)
        pe_data = {pe: len(instrs) for pe, instrs in self.pe_instructions.items()}
        pe_data = dict(sorted(pe_data.items()))
        bars = plt.bar(range(len(pe_data)), list(pe_data.values()))
        plt.xticks(range(len(pe_data)), list(pe_data.keys()))
        plt.title('PE Utilization')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 5. Instructions per Cycle
        plt.subplot(3, 2, 5)
        cycle_data = dict(sorted(self.cycle_instruction_counts.items()))
        plt.plot(list(cycle_data.keys()), list(cycle_data.values()), marker='o')
        plt.title('Instructions per Cycle')
        plt.xlabel('Cycle')
        plt.ylabel('Number of Instructions')
        
        # 6. Operation Type Distribution
        plt.subplot(3, 2, 6)
        op_data = dict(sorted(self.op_type_counts.items()))
        plt.pie(op_data.values(), labels=op_data.keys(), autopct='%1.1f%%')
        plt.title('Operation Type Distribution')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save with high DPI for better quality
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze CSL simulation log files.')
    parser.add_argument('--log_file', type=str, help='Path to the simulation log file')
    args = parser.parse_args()
    
    if not args.log_file:
        print("Error: Please provide a log file path using --log_file")
        sys.exit(1)
        
    analyzer = SimLogAnalyzer(args.log_file)
    analyzer.analyze()
    analyzer.print_summary()
    analyzer.plot_metrics()

if __name__ == "__main__":
    main() 