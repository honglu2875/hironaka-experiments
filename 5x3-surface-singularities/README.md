# Some checkpoint results (without fine-tuning, small-scale with max 5 points)
Here are the blowup trees given by a couple trained agents. But the hyperparameters are not fine-tuned and they are very small scale.

Recall that each integral point corresponds to a monomial, and we are considering the hypersurface defined by the sum of the monomial. For example, an initial state of $[[2,0,0], [0,2,0], [0,0,3]]$ corresponds to

$$x^2+y^2+z^3=0$$.

Feel free to check out the [best model](#model-with-014-score-against-choose-firstchoose-last). The rules of games are changed slightly (reposition the points after making the coordinate changes, so that we disregard the exceptional divisors). There are a lot of game states (existence of one of $[1, 0, 0], [0, 1, 0], [0, 0, 1]$ in the game state already implies smoothness). Thus, **the trees contain a lot of redundancy**.

# An example of the interpretation of the blowup tree
Let us take [D4](#d4-1) as an example. This resolution is one of the optimal gameplays for host, and it corresponds to the (unique) minimal resolution of D4 singularity (but not unique as an embedded resolution in $\mathbb A^3$):

$$x^2+y^2z+z^3=0.$$

Notice that the game initial state $[[2,0,0], [0,2,1], [0,0,3]]$ correspond to the exponent of $x,y,z$ of each monomial.

## Step 1, host move
The host chose the coordinates $[0, 2]$ as the first step, which corresponds to blowing up the line defined by $x=z=0$. The resulting surface is no longer an affine surface, and to look at the whole picture, one must try to observe through two different charts (think about atlas, charts in manifold theory):
- $(u,v,w)$ through the change of variables $u=x, v=y, uw=z$.
- $(u',v',w')$ through $u'w'=x, v'=y, w'=z$.
 
Choosing any chart corresponds to an agent action. And the change of variables corresponds to the linear transformations of the game states when looking at the exponents. This step is particularly surprising for me at a first glance, as the usual approach is to blow up $x=y=z=0$ (e.g., the first chart would be $u=x, uv=y, uw=z$, etc). But turns out it doesn't hurt the result.

A smart agent should choose the second chart, as the origin of the first chart is a already smooth point. Let us check this statement by hand: Plugging in $u=x, v=y, uw=z$, we obtain 

$$u^2+uv^2w+u^3w^3=u(u+v^2w+u^2w^3)=0.$$

The equation now defines two surfaces:
- $u=0$ which corresponds to the **exceptional divisor** of the blowup of $\mathbb A^3$. It is the "shadow" coming from the modification of the outer space $\mathbb A^3$, and spans outside the surface.
- $u+v^2w+u^2w^3=0$ which corresponds to the **strict transform** of the original surface. It is the real modification of the original surface, and it is what we care. **Exercise**: show that this is a smooth surface by calculating the Jacobian.

From the first chart, we can see that the **exceptional curves** consist of two lines: they are defined by $u+v^2w+u^2w^3=0, u=0$. By plugging $u=0$ in, the system of equation becomes $v^2w=0, u=0$, or equivalently, the line $u=v=0$ unions the line $u=w=0$.

## Step 1, agent move
The agent chose the second chart. Now by plugging in $u'w'=x, v'=y, w'=z$, we obtain $u'^2w'^2+v'^2w'+w'^3=w'(u'^2w'+v'^2+w'^2)=0$. Again,
- $w'=0$ is the exceptional divisor of the blowup on the ambient space $\mathbb A^3$.
- $u'^2w'+v'^2+w'^2=0$ is a singular surface. It glues together with $u+v^2w+u^2w^3=0$ and we are just seeing the two parts of the same surface.

(As a side note, the two exceptional divisors from different charts $u=0$ and $w'=0$ are not mutually exclusive. They glue together to form one single quasi-projective variety. We are also looking at two parts of the same exceptional divisor. As a result, the exceptional line $u=v=0$ and $w'=v'=0$ are in fact charts of the same $\mathbb P^1$.)

An easy application of Jacobian criterion tells us that the origin $(0,0,0)$ on $u'^2w'+v'^2+w'^2=0$ still needs to be resolved.

## Step 2, host move and agent move
Now we rinse and repeat from the new equation $u'^2w'+v'^2+w'^2=0$. The host chose all coordinates this time, which corresponds to blowing up at $u'=v'=w'=0$. With the analysis above as well as the help from the blowup tree, we see that only one chart is interesting (agent's choice of coordinate $0$). Altogether, they correspond to the change of variable:

$$u''=u', u''v''=v', u''w''=w'.$$

By plugging in, we see $u''^3w''+u''^2v''^2+u''^2w''^2=u''^2(u''w''+v''^2+w''^2)=0$. Ignoring the exceptional divisor $u''=0$ (for now), we move on to the next singular surface $u''w''+v''^2+w''^2=0$.

**Exercise:** finish the step 3.

Now, if we backtrace all the steps and keep track of the exceptional curves, passing to its dual graph, we will see the famous Dynkin diagram D4.

# Experiment record

## model with 0.1 score against choose-first/choose-last
### A1
![A1](0.1-model/A1.png)

A1 is the most classical nodal singularity. As long as it is not blowing up a line, it would be resolved with one single blowup.

### A2
![A2](0.1-model/A2.png)

A2 should have been resolved with one single blowup as well. But apparently the host made the wrong choice in the first step. However, a further trained checkpoint was able to find the simplest result.

### A3
![A3](0.1-model/A3.png)

A3 could be resolved in two blowups at 0-dimensional strata. Indeed the host found the best solution!

### D4
![D4](0.1-model/D4.png)

A resolution of D4 (this is perhaps slightly more complex than the D4 Dynkin diagram, but is it minimal under the constraint of blowing up toric strata?).

### D5
![D5](0.1-model/D5.png)

It did not terminate for D5 under prescribed maximal depth.

### E6
![E6](0.1-model/E6.png)

A nice resolution for E6.

### E7 and E8
[E7](0.1-model/E7.png) and [E8](0.1-model/E8.png) did not end up terminating.

## model with 0.14 score against choose-first/choose-last
Now in this case, we turned on `reposition`. After shifting, each coordinates will be translated so that at least one point touches the coordinate planes (the coordinate equals to zero).

In resolution, it is equivalent to the fact that we throw away the exceptional divisor, and only look at the strict transform (and make sure it is smoothed).

I emphasize that **the trees contain a lot of redundancy**, as I did not bake the following rule into the game: 
- containing a point $(x_1, x_2, x_3)$ with $x_1+x_2+x_3=1$ already implies smoothness.

### A2
![A2](0.14-model/A2.png)

A2. Optimal. In fact $[[1, 0, 0], [0, 0, 1]]$ was already smooth, but under our game it was not. Of course one could bake this into the rule of the game, but resolving the game from states like this seems to be easy enough. 

### A3
![A3](0.14-model/A3.png)

A3. Optimal

### A4
![A4](0.14-model/A4.png)

A4. Sadly the first choice was wrong. After the first blowup from $[[2,0,0], [0,2,0], [0,0,5]]$ to $[[2,0,2], [0,2,0], [0,0,5]]$, the A4 singularity did not improve at all (as they are locally isomorphic!). 

**Interesting question**: what is the singularity of $[[2,0,2], [0,2,0], [0,0,5]]$? 

From this step onwards, the tree does have the feeling of an optimal resolution but I am not sure.

### D4
![D4](0.14-model/D4.png)

D4. Optimal. One gets exactly the D4 Dynkin diagram from this tree (though not immediately obvious).
Note that any occurence of $(x_1, x_2, x_3)$ where $x_1+x_2+x_3=1$ is already a smooth point, despite the game not being terminate. We could tweak that in a later version.

Note that the last blowup (from $[[1,0,1], [0,2,0], [0,0,2]]$) in fact separates three lines intersecting at the singularity, thus achieving the final look of D4 Dynkin diagram.

### D5
![D5](0.14-model/D5.png)

D5. Almost optimal. From the state $[[2,0,0], [0,1,1], [0,0,2]]$, it would have obtained the D5 Dynkin diagram had the host chose all the coordinates $(0,1,2)$.

### E6
![E6](0.14-model/E6.png)

E6

### E7
![E7](0.14-model/E7.png)

E7

### E8
![E8](0.14-model/E8.png)

E8
