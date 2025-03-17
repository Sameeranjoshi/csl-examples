# High-level Python DSL to generate CSL dataflows

class Dataflow:
    def __init__(self, grid):
        self.grid = grid
        self.steps = []
        self.data_items = {}

    def data(self, name, size, initial_pe):
        self.data_items[name] = {'size': size, 'initial_pe': initial_pe}
        return self

    def scatter(self, data_name):
        step = {'action': 'scatter', 'data': data_name}
        self.steps.append(step)
        self.current_step = step
        return self

    def broadcast(self, data_name):
        step = {'action': 'broadcast', 'data': data_name}
        self.steps.append(step)
        self.current_step = step
        return self

    def compute(self, operation):
        self.steps.append({'action': 'compute', 'operation': operation})
        return self

    def reduce(self, axis, operation, direction):
        self.steps.append({
            'action': 'reduce',
            'axis': axis,
            'operation': operation,
            'direction': direction
        })
        return self

    def gather(self, axis, to_pe):
        self.steps.append({'action': 'gather', 'axis': axis, 'to_pe': to_pe})
        return self

    def across(self, direction):
        self.current_step['across'] = direction
        return self

    def from_pe(self, pe):
        self.current_step['from'] = pe
        return self

    def generate_csl(self, filename):
        with open(filename, 'w') as f:
            f.write('// Auto-generated CSL from DSL\n\n')
            f.write(f'layout {{ @set_rectangle({self.grid[0]}, {self.grid[1]}); }}\n\n')

            for data, props in self.data_items.items():
                f.write(f'// data {data} of size {props["size"]} at PE {props["initial_pe"]}\n')

            for step in self.steps:
                action = step['action']
                if action == 'scatter':
                    f.write(f'// scatter {step["data"]} across {step["across"]} from PE {step["from"]}\n')
                elif action == 'broadcast':
                    f.write(f'// broadcast {step["data"]} across {step["across"]} from PE {step["from"]}\n')
                elif action == 'compute':
                    f.write(f'// compute operation: {step["op"]}\n')
                elif action == 'reduce':
                    f.write(f'// reduce across {step["axis"]}, operation: {step["operation"]}, direction: {step["direction"]}\n')
                elif action == 'gather':
                    f.write(f'// gather across {step["axis"]} to PE {step["to_pe"]}\n')

# Usage Example
flow = Dataflow(grid=(10, 10))

flow.data('x', size=(10,1), initial_pe=(0,0))\
    .data('b', size=(10,1), initial_pe=(0,0))\
    .scatter('x').across('rows').from_pe((0,0))\
    .scatter('b').across('columns').from_pe((0,0))\
    .compute('Ax')\
    .reduce(axis='rows', operation='sum', direction='left')\
    .gather(axis='columns', to_pe=(0,0))\
    .scatter('residual').across('rows').from_pe((0,0))


flow.generate_csl('output.csl')

