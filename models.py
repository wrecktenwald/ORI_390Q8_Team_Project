"""
Implement general models for use in project
"""

import numpy as np
import pandas as pd
from pyomo.opt import SolverFactory
import pyomo.kernel as pmo
import pyomo.environ as pyo
import time

from utilities import elapsed_time_display


class model:
    """
    Parent model class

    Attributes
    __________
    periods : int
        Number of periods to run model for, must match length of period indexed arrays
    D : numpy.array
        Maximum discharge rate indexed by period as 1-D numpy.array
    C : numpy.array
        Maximum charge rate indexed by period as 1-D numpy.array
    S : numpy.array
        Storage capacity indexed by period as 1-D numpy.array
    p : numpy.array
        Energy price indexed by period as 1-D numpy.array 
    r : float
        Discount rate
    R : numpy.array or float
        Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
    vc : float
        Variable charge cost
    vd : float
        Variable discharge cost
    solver : str
        Solver to use in SolverFactory
    model : pyomo.environ.ConcreteModel or None
        Model
    results : return of pyomo.opt.SolverFactory or None
        Results of solve
    """
    def __init__(
        self,
        periods,
        D,
        C,
        S,
        p,
        r,
        R, 
        vc=0.0, 
        vd=0.0,
        solver='glpk'
    ):
        """
        Parameters
        __________
        periods : int
            Number of periods to run model for, must match length of period indexed arrays
        D : numpy.array
            Maximum discharge rate indexed by period as 1-D numpy.array
        C : numpy.array
            Maximum charge rate indexed by period as 1-D numpy.array
        S : numpy.array
            Storage capacity indexed by period as 1-D numpy.array
        p : numpy.array
            Energy price indexed by period as 1-D numpy.array 
        r : float
            Discount rate
        R : numpy.array or float
            Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
        vc : float, default 0.0
            Variable charge cost
        vd : float, default 0.0
            Variable discharge cost
        solver : str, default 'glpk'
            Solver to use in SolverFactory
        """

        self.periods = periods
        self.D = D
        self.C = C
        self.S = S
        self.p = p
        self.r = r
        self.R = R
        self.vc = vc
        self.vd = vd
        self.solver = solver

        self.model = None
        self.results = None
    
    def setup_model(
        self
    ):
        """
        Setup model
        """

        raise NotImplementedError

    def solve_model(
        self
    ):
        """
        Solve model
        """

        if self.model is None: 
            raise ValueError('Please setup model before solving')
        
        start = time.time()

        solver_factory = SolverFactory(self.solver)
        self.results = solver_factory.solve(self.model)
        print(f"""
        Solution Results for model
            STATUS: {self.results['Solver'][0]['Status'].value.upper()}
            TERMINATION CONDITION: {self.results['Solver'][0]['Termination condition'].value.upper()}
            OBJECTIVE LOWER BOUND: {self.results['Problem'][0]['Lower bound']}
            OBJECTIVE UPPER BOUND: {self.results['Problem'][0]['Upper bound']}
        """)
        self.model.solutions.store_to(self.results)  # store solutions too

        elapsed_time_display(start, ' (model solve time)')

    def write_to_json(
        self, 
        filename
    ):
        """
        Write results to a JSON

        Parameters
        __________
        filename : str
            Filename and location to save JSON, must end in '.json'
        """

        if self.results is None: 
            raise ValueError('Please use solve_model method before writing results as JSON')

        self.results.write(filename=filename, format='json')

    def optimal(self):
        self.setup_model()
        solver_factory = SolverFactory(self.solver)
        self.results = solver_factory.solve(self.model)
        return self.results['Problem'][0]['Lower bound']

class proposal_v2_1(model):
    """
    Updated model detailed in theory in Team Project Proposal Follow-up v2.docx

    Attributes
    __________
    valid_pair_span : int
        Valid maximum span between pairs of periods measured in periods
    periods : int
        Number of periods to run model for, must match length of period indexed arrays
    D : numpy.array
        Maximum discharge rate indexed by period as 1-D numpy.array
    C : numpy.array
        Maximum charge rate indexed by period as 1-D numpy.array
    S : numpy.array
        Storage capacity indexed by period as 1-D numpy.array
    p : numpy.array
        Energy price indexed by period as 1-D numpy.array 
    r : float
        Discount rate
    R : numpy.array or float
        Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
    vc : float
        Variable charge cost
    vd : float
        Variable discharge cost
    solver : str
        Solver to use in SolverFactory
    model : pyomo.environ.ConcreteModel or None
        Model
    results : return of pyomo.opt.SolverFactory or None
        Results of solve
    """
    def __init__(
        self,
        valid_pair_span,
        periods,
        D,
        C,
        S,
        p,
        r,
        R, 
        vc=0.0, 
        vd=0.0,
        solver='glpk'
    ):
        """
        Parameters
        __________
        valid_pair_span : int
            Valid maximum span between pairs of periods measured in periods
        periods : int
            Number of periods to run model for, must match length of period indexed arrays
        D : numpy.array
            Maximum discharge rate indexed by period as 1-D numpy.array
        C : numpy.array
            Maximum charge rate indexed by period as 1-D numpy.array
        S : numpy.array
            Storage capacity indexed by period as 1-D numpy.array
        p : numpy.array
            Energy price indexed by period as 1-D numpy.array 
        r : float
            Discount rate
        R : numpy.array or float
            Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
        vc : float, default 0.0
            Variable charge cost
        vd : float, default 0.0
            Variable discharge cost
        solver : str, default 'glpk'
            Solver to use in SolverFactory
        """

        if isinstance(R, float):
            R = np.full((periods, periods), R)  # will only reference valid periods
        super().__init__(periods, D, C, S, p, r, R, vc, vd, solver)

        self.valid_pair_span = valid_pair_span

    def setup_model(
        self
    ):
        """
        Setup model
        """

        start = time.time()

        T = np.arange(0, self.periods)  # use native 0 indexing in Python
        V = [(_, _ + span) for _ in T for span in np.arange(1, self.valid_pair_span + 1) if _ + span < self.periods]  # only consider valid periods
        
        self.model = pyo.ConcreteModel()
        self.model.constraints = pyo.ConstraintList()
        self.model.f = pyo.Var(V, domain=pyo.NonNegativeReals)
        self.model.s = pyo.Var(T, domain=pyo.NonNegativeReals)

        self.model.objective = pyo.Objective(
            expr=sum(
                (self.model.f[idx] * ((self.p[idx[1]] - self.vd) * self.R[idx[0]][idx[1]] - (self.p[idx[0]] + self.vc))) *  # pair arb
                (np.exp(-self.r * idx[1] / self.periods)) for idx in V),  # discounting  
            sense=pyo.maximize
        )

        for pd in T:  # TODO: can for loop be vectorized, otherwise streamlined
            self.model.constraints.add(self.model.s[pd] <= self.S[pd])
            self.model.constraints.add(self.model.s[pd] == sum(self.model.f[idx] for idx in V if ((idx[0] <= pd) & (idx[1] > pd))))
            try: 
                self.model.constraints.add(sum(self.model.f[idx] for idx in V if (idx[1] == pd)) <= self.D[pd])
            except ValueError:  # no valid periods
                pass
            try: 
                self.model.constraints.add(sum(self.model.f[idx] for idx in V if (idx[0] == pd)) <= self.C[pd])    
            except ValueError:  # no valid periods
                pass

        elapsed_time_display(start, ' (model setup time)')


class linear_model_v1_0(model):
    """
    Efficient model detailed by Minh with separated charge, discharge periods

    Attributes
    __________
    max_charge : int
        Maximum number of batch charges allowed
    batch_runtime : int
        Number of periods it takes to charge (no discharge allowed)
    periods : int
        Number of periods to run model for, must match length of period indexed arrays
    D : numpy.array
        Maximum discharge rate indexed by period as 1-D numpy.array
    C : numpy.array
        Maximum charge rate indexed by period as 1-D numpy.array
    S : numpy.array
        Storage capacity indexed by period as 1-D numpy.array
    p : numpy.array
        Energy price indexed by period as 1-D numpy.array 
    r : float
        Discount rate
    R : numpy.array or float
        Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
    vc : float
        Variable charge cost
    vd : float
        Variable discharge cost
    solver : str
        Solver to use in SolverFactory
    model : pyomo.environ.ConcreteModel or None
        Model
    results : return of pyomo.opt.SolverFactory or None
        Results of solve
    """
    def __init__(
        self,
        max_charge, 
        batch_runtime, 
        periods,
        D,
        C,
        S,
        p,
        r,
        R, 
        vc=0.0, 
        vd=0.0,
        solver='glpk'
    ):
        """
        Parameters
        __________
        max_charge : int
            Maximum number of batch charges allowed
        batch_runtime : int
            Number of periods it takes to charge (no discharge allowed)
        periods : int
            Number of periods to run model for, must match length of period indexed arrays
        D : numpy.array
            Maximum discharge rate indexed by period as 1-D numpy.array
        C : numpy.array
            Maximum charge rate indexed by period as 1-D numpy.array
        S : numpy.array
            Storage capacity indexed by period as 1-D numpy.array
        p : numpy.array
            Energy price indexed by period as 1-D numpy.array 
        r : float
            Discount rate
        R : numpy.array or float
            Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
        vc : float, default 0.0
            Variable charge cost
        vd : float, default 0.0
            Variable discharge cost
        solver : str, default 'glpk'
            Solver to use in SolverFactory
        """

        if isinstance(R, np.ndarray):
            raise ValueError('This model does not accept matrix or numpy.array R input')
        super().__init__(periods, D, C, S, p, r, R, vc, vd, solver)

        self.max_charge = max_charge
        self.batch_runtime = batch_runtime

    def setup_model(
        self
    ):
        """
        Setup model
        """

        start = time.time()

        T = np.arange(0, self.periods)  # use native 0 indexing in Python
        big_M = float('inf')
        
        self.model = pyo.ConcreteModel()
        self.model.constraints = pyo.ConstraintList()
        self.model.d = pyo.Var(T, domain=pyo.NonNegativeReals) # Discharge
        self.model.c = pyo.Var(T, domain=pyo.NonNegativeReals) # Charge
        self.model.s = pyo.Var(T, domain=pyo.NonNegativeReals) # Current storage
        self.model.charging = pyo.Var(T, domain=pyo.Binary) # Binary variable indicating whether batch charging is in progress

        self.model.objective = pyo.Objective(
            expr=sum(
                ((self.p[t] - self.vd) * self.R * self.model.d[t] - (self.p[t] + self.vc) * self.model.c[t]) *   # pair arb
                (np.exp(-self.r * t / self.periods)) for t in T),  # discounting   
            sense=pyo.maximize
        )

        self.model.constraints.add(self.model.s[0] == 0) # initial storage set to 0
        self.model.constraints.add(sum(self.model.charging[t] for t in T) <= (self.max_charge * self.batch_runtime))
        for pd in T:  # TODO: can for loop be vectorized, otherwise streamlined
            self.model.constraints.add(self.model.s[pd] <= self.S[pd])
            self.model.constraints.add(self.model.d[pd] <= self.D[pd])
            self.model.constraints.add(self.model.d[pd] <= self.model.s[pd])  # discharge cannot exceed what's currently in storage
            self.model.constraints.add(self.model.c[pd] <= self.C[pd] * self.model.charging[pd])  # if not charging then c[t] is zero

            if pd > 0:
                self.model.constraints.add(self.model.s[pd-1] + self.model.c[pd] - self.model.d[pd] == self.model.s[pd])  # storage equation

                # The following constraints ensure that batch charging lasts exactly batch_runtime
                # Refer to: https://yetanothermathprogrammingconsultant.blogspot.com/2018/03/production-scheduling-minimum-up-time.html
                self.model.constraints.add(sum([self.model.charging[k] for k in range(pd, min(self.periods, pd + self.batch_runtime))]) >= self.batch_runtime * (self.model.charging[pd] - self.model.charging[pd - 1]))   
                self.model.constraints.add(sum([self.model.charging[k] for k in range(pd, min(self.periods, pd + self.batch_runtime + 1))]) <= self.batch_runtime)
            else:
                self.model.constraints.add(sum([self.model.charging[k] for k in range(0, min(self.periods, self.batch_runtime))]) >= self.batch_runtime * (self.model.charging[0]))  
            
            self.model.constraints.add(self.model.d[pd] <= big_M * (1 - self.model.charging[pd])) # cannot discharge while charging

        elapsed_time_display(start, ' (model setup time)')


class vp_v3_0(model):
    """
    Updated model detailed in theory in Team Project Proposal Follow-up v2.docx based on the concept of valid pairs with addition of batch charge, discharge runtime

    Attributes
    __________
    valid_pair_span : int
        Valid maximum span between pairs of periods measured in periods
    batch_c_rt : int
        Number of consecutive periods it takes to charge with no discharge allowed
    batch_d_rt : int
        Number of consecutive periods it takes to discharge with no charge allowed
    periods : int
        Number of periods to run model for, must match length of period indexed arrays
    D : numpy.array
        Maximum discharge rate indexed by period as 1-D numpy.array
    C : numpy.array
        Maximum charge rate indexed by period as 1-D numpy.array
    S : numpy.array
        Storage capacity indexed by period as 1-D numpy.array
    p : numpy.array
        Energy price indexed by period as 1-D numpy.array 
    r : float
        Discount rate
    R : numpy.array or float
        Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
    vc : float
        Variable charge cost
    vd : float
        Variable discharge cost
    solver : str
        Solver to use in SolverFactory
    model : pyomo.environ.ConcreteModel or None
        Model
    results : return of pyomo.opt.SolverFactory or None
        Results of solve
    """
    def __init__(
        self,
        valid_pair_span,
        periods,
        D,
        C,
        S,
        p,
        r,
        R, 
        batch_c_rt=None,
        batch_d_rt=None,
        vc=0.0, 
        vd=0.0,
        solver='glpk'
    ):
        """
        Parameters
        __________
        valid_pair_span : int
            Valid maximum span between pairs of periods measured in periods
        periods : int
            Number of periods to run model for, must match length of period indexed arrays
        D : numpy.array
            Maximum discharge rate indexed by period as 1-D numpy.array
        C : numpy.array
            Maximum charge rate indexed by period as 1-D numpy.array
        S : numpy.array
            Storage capacity indexed by period as 1-D numpy.array
        p : numpy.array
            Energy price indexed by period as 1-D numpy.array 
        r : float
            Discount rate
        R : numpy.array or float
            Round trip efficiency matrix time dependent on charge (i), discharge (j) periods or non time dependent
        batch_c_rt : int, default None
            Number of consecutive periods it takes to charge with no discharge allowed
        batch_d_rt : int, default None
            Number of consecutive periods it takes to discharge with no charge allowed
        vc : float, default 0.0
            Variable charge cost
        vd : float, default 0.0
            Variable discharge cost
        solver : str, default 'glpk'
            Solver to use in SolverFactory
        """

        if isinstance(R, float):
            R = np.full((periods, periods), R)  # will only reference valid periods
        super().__init__(periods, D, C, S, p, r, R, vc, vd, solver)

        self.valid_pair_span = valid_pair_span
        self.batch_c_rt = batch_c_rt
        self.batch_d_rt = batch_d_rt

    def setup_model(
        self
    ):
        """
        Setup model
        """

        start = time.time()

        T = np.arange(0, self.periods)  # use native 0 indexing in Python
        V = [(_, _ + span) for _ in T for span in np.arange(1, self.valid_pair_span + 1) if _ + span < self.periods]  # only consider valid periods
        
        self.model = pyo.ConcreteModel()
        self.model.constraints = pyo.ConstraintList()
        self.model.f = pyo.Var(V, domain=pyo.NonNegativeReals)
        self.model.s = pyo.Var(T, domain=pyo.NonNegativeReals)
        if self.batch_c_rt:
            self.model.c = pyo.Var(T, domain=pyo.Binary) # Binary variable indicating whether batch charging is in progress
        if self.batch_d_rt:
            self.model.d = pyo.Var(T, domain=pyo.Binary) # Binary variable indicating whether batch charging is in progress

        self.model.objective = pyo.Objective(
            expr=sum(
                (self.model.f[idx] * ((self.p[idx[1]] - self.vd) * self.R[idx[0]][idx[1]] - (self.p[idx[0]] + self.vc))) *  # pair arb
                (np.exp(-self.r * idx[1] / self.periods)) for idx in V),  # discounting  
            sense=pyo.maximize
        )

        self.model.constraints.add(self.model.s[0] == 0)  # initial storage set to 0, already forced by valid period gen, reinforce for MILP
        for pd in T:  # TODO: can for loop be vectorized, otherwise streamlined

            if pd > 0:
                self.model.constraints.add(self.model.s[pd] <= self.S[pd])
                self.model.constraints.add(self.model.s[pd] == sum(self.model.f[idx] for idx in V if ((idx[0] <= pd) & (idx[1] > pd))))
            
            try:
                if self.batch_c_rt:
                    self.model.constraints.add(sum(self.model.f[idx] for idx in V if (idx[1] == pd)) <= (self.D[pd] * (1 - self.model.c[pd])))  # tighter bound than big M, no discharge while charging
                else: 
                    self.model.constraints.add(sum(self.model.f[idx] for idx in V if (idx[1] == pd)) <= self.D[pd])
            except ValueError:  # no valid periods
                pass
            try: 
                if self.batch_d_rt:
                    self.model.constraints.add(sum(self.model.f[idx] for idx in V if (idx[0] == pd)) <= (self.C[pd] * (1 - self.model.d[pd])))  # tighter bound than big M, no charge while discharging    
                else: 
                    self.model.constraints.add(sum(self.model.f[idx] for idx in V if (idx[0] == pd)) <= self.C[pd])
            except ValueError:  # no valid periods
                pass
            
            if self.batch_c_rt:
                if pd > 0:
                    self.model.constraints.add(sum([self.model.c[_] for _ in range(pd, min(self.periods, pd + self.batch_c_rt))]) >= self.batch_c_rt * (self.model.c[pd] - self.model.c[pd - 1]))   
                    self.model.constraints.add(sum([self.model.c[_] for _ in range(pd, min(self.periods, pd + self.batch_c_rt + 1))]) <= self.batch_c_rt)
                else:
                    self.model.constraints.add(sum([self.model.c[_] for _ in range(0, min(self.periods, self.batch_c_rt))]) >= self.batch_c_rt * (self.model.c[0]))  

            if self.batch_d_rt:
                if pd > 0:
                    self.model.constraints.add(sum([self.model.d[_] for _ in range(pd, min(self.periods, pd + self.batch_d_rt))]) >= self.batch_d_rt * (self.model.d[pd] - self.model.d[pd - 1]))   
                    self.model.constraints.add(sum([self.model.d[_] for _ in range(pd, min(self.periods, pd + self.batch_d_rt + 1))]) <= self.batch_d_rt)
                else:
                    self.model.constraints.add(sum([self.model.d[_] for _ in range(0, min(self.periods, self.batch_d_rt))]) >= self.batch_d_rt * (self.model.d[0]))  

        elapsed_time_display(start, ' (model setup time)')
