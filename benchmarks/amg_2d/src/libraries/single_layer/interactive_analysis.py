#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

class InteractiveSimLogAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.instructions = []
        self.pe_instructions = defaultdict(list)
        self.instruction_counts = defaultdict(int)
        self.cycle_instructions = defaultdict(list)
        self.pipe_usage = defaultdict(int)
        self.task_instructions = defaultdict(int)
        self.op_type_counts = defaultdict(int)
        self.cycle_instruction_counts = defaultdict(int)
        
    def parse_line(self, line: str) -> Tuple[dict, bool]:
        """Parse a single line of the sim log."""
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
                    
                self.instructions.append(data)
                self.pe_instructions[f"P{data['pe']}.{data['pe_sub']}"].append(data)
                self.instruction_counts[data['instruction']] += 1
                self.cycle_instructions[data['cycle']].append(data)
                self.pipe_usage[f"Pipe{data['pipe']}"] += 1
                self.task_instructions[f"T{data['task']}"] += 1
                self.op_type_counts[data['op_type']] += 1
                self.cycle_instruction_counts[data['cycle']] += 1

    def create_interactive_dashboard(self):
        """Create an interactive dashboard using Plotly."""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Instruction Distribution',
                'Pipe Usage',
                'Task Distribution',
                'PE Utilization',
                'Instructions per Cycle',
                'Operation Type Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}]
            ]
        )

        # 1. Instruction Distribution
        instr_data = dict(sorted(self.instruction_counts.items(), key=lambda x: x[1], reverse=True))
        fig.add_trace(
            go.Bar(
                x=list(instr_data.keys()),
                y=list(instr_data.values()),
                text=list(instr_data.values()),
                textposition='auto',
                name='Instructions'
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text="Instruction Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

        # 2. Pipe Usage
        pipe_data = dict(sorted(self.pipe_usage.items()))
        fig.add_trace(
            go.Bar(
                x=list(pipe_data.keys()),
                y=list(pipe_data.values()),
                text=list(pipe_data.values()),
                textposition='auto',
                name='Pipe Usage'
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Pipe", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        # 3. Task Distribution
        task_data = dict(sorted(self.task_instructions.items(), key=lambda x: x[1], reverse=True))
        fig.add_trace(
            go.Bar(
                x=list(task_data.keys()),
                y=list(task_data.values()),
                text=list(task_data.values()),
                textposition='auto',
                name='Tasks'
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Task ID", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        # 4. PE Utilization
        pe_data = {pe: len(instrs) for pe, instrs in self.pe_instructions.items()}
        pe_data = dict(sorted(pe_data.items()))
        fig.add_trace(
            go.Bar(
                x=list(pe_data.keys()),
                y=list(pe_data.values()),
                text=list(pe_data.values()),
                textposition='auto',
                name='PE Usage'
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text="PE", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        # 5. Instructions per Cycle
        cycle_data = dict(sorted(self.cycle_instruction_counts.items()))
        fig.add_trace(
            go.Scatter(
                x=list(cycle_data.keys()),
                y=list(cycle_data.values()),
                mode='lines+markers',
                name='Instructions/Cycle'
            ),
            row=3, col=1
        )
        fig.update_xaxes(title_text="Cycle", row=3, col=1)
        fig.update_yaxes(title_text="Instructions", row=3, col=1)

        # 6. Operation Type Distribution
        op_data = dict(sorted(self.op_type_counts.items()))
        fig.add_trace(
            go.Pie(
                labels=list(op_data.keys()),
                values=list(op_data.values()),
                textinfo='label+percent+value',
                name='Operation Types'
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="CSL Simulation Analysis Dashboard",
            showlegend=False,
            template="plotly_white"
        )

        # Save as HTML file for interactive viewing
        fig.write_html("simulation_dashboard.html")
        
        # Also save as static PNG for reference
        fig.write_image("simulation_dashboard.png")

def main():
    parser = argparse.ArgumentParser(description='Create interactive analysis dashboard for CSL simulation logs.')
    parser.add_argument('--log_file', type=str, help='Path to the simulation log file')
    args = parser.parse_args()
    
    if not args.log_file:
        print("Error: Please provide a log file path using --log_file")
        sys.exit(1)
        
    analyzer = InteractiveSimLogAnalyzer(args.log_file)
    analyzer.analyze()
    analyzer.create_interactive_dashboard()
    print("\nDashboard generated! Open 'simulation_dashboard.html' in your web browser to view the interactive analysis.")
    print("A static PNG version has also been saved as 'simulation_dashboard.png'")

if __name__ == "__main__":
    main()