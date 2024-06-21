import argparse
import numpy as np

from my_types import existing_path

class Grid(object):

    def __init__(self, value_0s: list, x_min: float = -1, x_max: float = 1,
                 end_points = [0, 0]):
        self.delta_x = (x_max - x_min) / (len(value_0s)) 
        self.size = len(value_0s) + 2
        self.xs = np.linspace(x_min - 0.5 * self.delta_x, x_max + 0.5 * self.delta_x, len(value_0s) + 2, endpoint=True)[:]
        print(self.xs)
        print(self.delta_x)
        self.values = np.array([end_points[0]] + list(value_0s) + [end_points[1]])
        self.x_min = x_min
        self.x_max = x_max
    
    def approx_partials(self) -> np.array:
        """
        Approximates the partial derivate {partial y}/{partial x}
        on a grid with grid spacing delta_x.
        
        Returns:
            _ (np.array): Array of 1D partial derivatives of values with respect to x.
        """
        dys = np.empty(self.values.size)
        dys[0] = 0
        dys[1:-1] = (self.values[2:] - self.values[:-2])
        dys[-1] = 0
        return dys / (2. * self.delta_x)

    @classmethod
    def read_from_file(cls, filepath: str, line_num: int = 0, delimiter: str=" ", **kwargs):
        line = read_file_line_number(filepath, line_num)
        t, *values = line.split(delimiter)
        t = float(t)
        values = np.array(values, dtype=float)
        return t, cls(values, **kwargs)
    
    def copy(self):
        return Grid(self.values, self.x_min, self.x_max)
        
    def __str__(self):
        return " ".join(list(self.values))


class Density(Grid):
    
    def __init__(self, value_0s: np.array, x_min: float = -1, x_max: float = 1,
                 epsilon: float = 1):
        super().__init__(value_0s, x_min, x_max)
        self.epsilon = epsilon
        self.neighborhood = int(np.floor(epsilon / self.delta_x))

    def copy(self):
        return Density(self.values[1:-1], self.x_min, self.x_max, self.epsilon)
    
    def update(self, rho_0s, rs, pr_po: np.array, dt: float):
        """
        Integrates rho over the time step dt at the xs using Euler's method.

        Args:
            rho_0s (Density): Initial density.
            pr_po (np.array): Array of partial deriviatives of the rate.
            dt (float):  Time step of integration.
        """
        prho_po = rho_0s.approx_partials()
        #print(rs.values)
        #print(rho_0s.values)
        #print(pr_po)
        #print(prho_po)
        #print(dt)
        self.values[1: -1] = rho_0s.values[1: -1] - (pr_po[1: -1] * rho_0s.values[1: -1] + rs.values[1: -1] * prho_po[1: -1]) * dt
        return self.values

    def neighborhood_means(self) -> np.array:
        """
        Returns the neighborhood means of the density.
        
        Returns:
            mean_xs (np.array): The neighborhood means. 
        """
    
        mean_xs = np.empty(self.size)
        min_i = -self.neighborhood
        max_i = self.neighborhood
        sum_pxs = (self.values[0: max_i] * self.xs[0: max_i]).sum() + 0.5 * self.values[max_i] * self.xs[max_i]  
        norm = self.values[0: max_i].sum() + 0.5 * self.values[max_i]
        mean_xs[0] = sum_pxs / norm
        for i in range(1, self.size):
            min_i += 1
            max_i += 1
            delta_sum = 0
            delta_norm = 0
            if min_i >= 0:
                delta_sum -= 0.5*self.values[min_i] * self.xs[min_i]
                norm -= 0.5 * self.values[min_i]
            if min_i > 0:
                delta_sum -= 0.5*self.values[min_i-1] * self.xs[min_i-1]
                norm -= 0.5*self.values[min_i-1] 
            if max_i <= self.size:
                delta_sum += 0.5*self.values[max_i-1] * self.xs[max_i-1]
                norm += 0.5*self.values[max_i-1]
            if max_i < self.size:
                delta_sum += 0.5*self.values[max_i] * self.xs[max_i]
                norm += 0.5*self.values[max_i]
                
            sum_pxs += delta_sum
            norm += delta_norm
            mean_xs[i] = sum_pxs / norm
        return mean_xs   
    
class Rate(Grid):
    
    def __init__(self, rho: Density):
        super().__init__(np.empty(rho.size - 2), rho.x_min, rho.x_max)
        self.HK_dynamics(rho)
    
    def HK_dynamics(self, rho: Density):
        """
        Update the rate using the input density according to HK dynamics.

        Args:
            rho (Density): Density defined on the x-grid. 
            alphas (np.array, optional): Weights applied to rates. Defaults to None.
        """
        mean_xs = rho.neighborhood_means()
        # Apply the HK-dynamics equation
        self.values = mean_xs - rho.xs
        self.values[np.where(np.abs(self.values) < 1e-12)] = 0     
        
        
class FileReporter(object):
    
    def __init__(self, filepath: str, every: int, dt: float, t: float=0):
        self.every = every
        self.filepath = filepath
        self.fh = open(self.filepath, "w")
        self.first_line = True
        self.count = 0
        self.t = t
        self.dt = dt
    
    def increment(self):
        self.count += 1
        self.t += self.dt
    
    def should_write(self):
        return self.count % self.every == 0
    
    def write(self, grid_in: Grid):
        if not self.first_line:
            self.fh.write("\n")
            self.first_line = False
        self.fh.write(str(self.t) + " ")
        self.fh.write(" ".join([str(v) for v in grid_in.values]))
        self.fh.write("\n")
        print(grid_in.values.sum()*grid_in.delta_x)
            
    def close(self):
        self.fh.close()


def read_file_line_number(filepath: str, line_num: int) -> str:
    with open(filepath, "r") as fh:
        count = 0
        for line in fh:
            if count == line_num:
                return line.strip()
            count += 1
    err_msg = "Line number, " + str(line_num) + ", is beyond the end of " + filepath + "."
    raise ValueError(err_msg)

    
def main():
    """
    Solve the trajectory integration of the density using Euler.
    """
    args = add_and_parser_cli(main.__doc__)
    
    # Read in initial density and set up for recursion.
    t, rho = Density.read_from_file(args.rho_filepath, args.line_num, x_min=args.x_min,
                                    x_max=args.x_max, epsilon=args.epsilon)
    prev_rho = rho.copy()
    # Determine initial rates.
    rs = Rate(rho)
    
    pr_pos = rs.approx_partials()
    
    # Set up incremental reporter.
    reporter = FileReporter(args.output_file, args.every, args.dt, t)
    print(rho.values.sum() * rho.delta_x)

    # Iterate until done.
    while reporter.count < int(args.num_dt):
        # Update the density
        rho.update(prev_rho, rs, pr_pos, reporter.dt)
        
        # Handle incremental reporter.
        reporter.increment()
        if reporter.should_write():
            reporter.write(rho)
        
        # Determine current rates.
        rs.HK_dynamics(rho)
        
        # Save current density to previous.
        prev_rho.values = rho.values
        

def add_and_parser_cli(description: str) -> argparse.Namespace:
    """
    Isolates the command line interface.

    Args:
        description (str): String passed to the help of the parser.

    Returns:
        args (argparse.Namespace): Namespace object with the values stored. 
    """
    
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument("rho_filepath", type=existing_path, help="Initial density file.")
    parser.add_argument("dt", type=float, help="Integration stepsize.")
    parser.add_argument("num_dt", type=float, help="The total number of time steps to evolve.")
    parser.add_argument("epsilon", type=float, help="Defines the neighborhood.")
    parser.add_argument("output_file", type=str, help="Where the output densities will be stored.")
    parser.add_argument("-l", "--line_num", dest="line_num", type=int, default=0,
                        help="The rho_filpath line number to use for the initial 11density."
                        + "  Default is 0 (the first line).")
    parser.add_argument("--x_min", type=float, default=-1, help="The minimum of the x-grid."
                        + "  Defaults to -1.")
    parser.add_argument("--x_max", type=float, default=1, help="The maximum of the x-grid."
                        + "  Defaults to 1.")
    parser.add_argument("-e", "--every", dest="every", type=int, default=1,
                        help="When to report."
                        + "  Default is 1 (every density).")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()