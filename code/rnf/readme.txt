recurrent neural field


idea:
  have nf: neural field
         : R^n -> R^c
  have control: neural network
              : R^{H1, H2, ..., Hn, C} -> Endomorphisms(R^n)

  now make a meshgrid, sample the network, pass into control, use endomorphism

  as a pushforward for mesh, and now update the neural field to enforce invariance or equivariance.




