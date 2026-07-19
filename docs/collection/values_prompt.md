# Values for this codebase.

Here I describe we should seriously re-evaluate everything about this codebase to make it more maintainable and easier for claude to read in the future.


## Overall Structure and Conceptual Entanglements

The code must be highly decoupled and well structure must be well defined beforehand.

We can use graphify to find connections between code and identify areas that need to be focused on.

Names should be defined properly. If any name is used to refer to some piece of content in the codebase. This should be well defined. This is why we previously undid a lot of confusion through both synonyms and distinctions in names that hinted towards different implementations (e.g. special cases for some), even though all implementations should have been the same, and they should have been the same item. The problem we previously had was that we had the words node, leaf, factor, hook, signal, and more, that essentially referred to the same thing. We still have some of this confusion lying around, this problem has mostly been eliminated functionally, but there are still references to words that should be called something else, so this problem hasn't been eliminated yet. The longer this stays, the more likely agents will think "oh that's something different, I should make this again for this item" even though it is in reality the same, but with a different name.

We also have problems with files and classes having vague responsibilities. I am still honestly quite confused why we have 3 files for hook stuff. If it was necessary, I need a real good understanding of why.

To help with the decoupling of the codebase, I'd like to build a proper skeleton in isolation that documents the interfaces of each class (maybe using abstract base classes? Are there better tools to do something like this?). And then analyse where each is used and build the minimum coupling from the ground up. Perhaps there is a better way of analysing coupling and redesigning the codebase to work correctly with this.

Besides inter-file problems, which are of course also an audit priority, an intra-file example of a badly structured, out-of-place, special case mess is the set of systems `project_same_layer`, `project_onto_basis`, and `set_token_filter`.

```python
def _project_same_layer_worker(args):
    in_path, out_path, onto = args
    return DataAccessor(in_path).project_same_layer(onto=onto, output_path=out_path)

def _project_onto_file_worker(args):
    in_path, out_path, basis_path = args
    return DataAccessor(in_path).project_onto_basis(basis_path, output_path=out_path)

def _set_filter_worker(args):
    in_path, out_path, filter_dict = args
    return DataAccessor(in_path).set_token_filter(filter_dict, output_path=out_path)
```

Each are dedicated vertical processes with their own workers as shown above, doing every operation from head to toe as a non-generalisable operation unique to only themselves. Even though that operation might very well be needed as a more simple feature by other components. In fact, the set_token_filter is so much of a joke, that it's implementing something for itself:

```python
    def set_token_filter(self, filter_dict: dict, output_path: str = None) -> str:
        self.data["__token_filter__"] = filter_dict
        return self._save_in_place(output_path)
```

Even though setting a token filter is literally a built in feature of the general save function that any standard usage of the class uses:

```python
    def save(self, path, format="cov_svd", cross_basis_refs=None,
             storage_dtype=None, token_filter=None, n_chunks=None):
        result = self.to_dict(format=format, storage_dtype=storage_dtype)

        if cross_basis_refs:
            target = DataAccessor(result, model=self.model, model_config=self.model_config,
                                  model_name=self._model_name)
            for ref_path in cross_basis_refs:
                target._project_onto(ref_path)

        self._stamp_from_path(path)
        self._write_metadata(result, format, token_filter=token_filter, n_chunks=n_chunks)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(result, path)
        return path
```

There is literally no point in setting call set_token_filter, you literally just have to call save with the `token_filter` argument!

In fact, the `_save_in_place` function:
```python
    def _save_in_place(self, output_path: str = None) -> str:
        out = output_path or self.path
        if out is None:
            raise ValueError("output_path is required for in-memory data")
        self._stamp_from_path(out)
        self._write_metadata(self.data, self.data.get("__format__", "unknown"))
        torch.save(self.data, out)
        return out
```

Is _solely_ used by the above functions I mentioned.
Although it is true that the current save function asks for a destination format and defaults to something other than the format the data is already in, I think that seems like a somewhat unnecessary default. Any data already would or should have it's format known, since the only thing saving this data is already using the save function. In fact, the only threat _to_ that is the fact that there is _this_ secondary save function that accepts a format not being known. It may also be worthwhile to guarantee that save the is not able to write anything differently than its own input except for making sure metadata is correct. (Iirc if we don't by default change the storage format, the policy is already this, but we should make sure).

Now let's look more into the projections:

```python
    def project_same_layer(self, onto: str = "both", output_path: str = None) -> str:
        """Cross-project acts and grads onto each other's basis at every leaf that
        carries both in the same space (boundary / slice / a projection's .out)."""
        do_fwd = onto in ("fwd", "both")
        do_rev = onto in ("rev", "both")
        for leaf in self._all_leaves():
            a = self.factor(leaf, "acts")
            g = self.factor(leaf, "grads")
            if a is None or g is None:
                continue
            av, gv, ac, gc = a.eigvecs, g.eigvecs, a.cov, g.cov
            if any(x is None for x in (av, gv, ac, gc)) or av.shape != gv.shape:
                continue
            e = self.data.setdefault(self._canon(leaf), {})
            if do_fwd:
                e["grads_cross_eigvals_acts"] = cross_eigvals(gc, av)
            if do_rev:
                e["acts_cross_eigvals_grads"] = cross_eigvals(ac, gv)
        return self._save_in_place(output_path)

    def project_onto_basis(self, basis_path: str, output_path: str = None) -> str:
        self._project_onto(basis_path)
        return self._save_in_place(output_path)

    def _project_onto(self, basis_path: str) -> None:
        ref = DataAccessor(basis_path)
        label = os.path.splitext(os.path.basename(basis_path))[0]
        for leaf in self._all_leaves():
            for q in ("acts", "grads"):
                a = self.factor(leaf, q)
                b = ref.factor(leaf, q)
                if a is None or b is None:
                    continue
                basis, cov = b.eigvecs, a.cov
                if basis is None or cov is None or cov.shape[0] != basis.shape[0]:
                    continue
                self.data.setdefault(self._canon(leaf), {})[f"{q}_cross_eigvals_{label}"] = \
                    cross_eigvals(cov, basis)
```

So we see that `project_onto_basis` is... a wrapper that calls `_project_onto`, and then saves it. You know what else can call project onto, and then save it? That's right! THE SAVE FUNCTION.

I should also add that both `project_onto_basis` and `_project_onto` have horrible names, since they are only one very specific case of project onto something, they only project onto the same hook point of a different file.

Now finally, we come to `project_same_layer`. Something so unique to itself that it honestly does not deserve to exist. How on earth are we creating a system to project different points of this checkpoint onto each other, but it's _only_ allowed to be used by this one process? That's just absurd. If we do support this feature, it should just be a normal part of the data accessor like any other. If we are going to build a projection feature, it should be general such that you can call a projection between any two points in the model. A feature that creates extra points specifically for projections between acts and gradients should then allowed to call one leaf's acts and another leaf's acts. It is however, also potentially better if such extra keys on a leaf node simply _don't exist_, because it breaks a few assumptions about what we could find in a leaf node. I do however think that Accessor should allow a user to call to view any quantity through any projection. The implementation of this is something that should be discussed. How this ties into compute_metrics is genuinely arguable, and I think that for now we should just scrap `project_same_layer`, and then decide on how we implement projection.


Now let's bring up an example of an inter-file complication.

There is a severe confusion of where and where not numpy or pytorch data formats and calls are allowed to be used. Deciding on strict rules for this would greatly help avoid confusion and bugs.

In general, we prefer to use pytorch for all kinds of computation. Basically anything should be done on the GPU if possible. However, for the experiments notebook, we'd like to use numpy because numpy is a light-weight easy to use library with less dependency, making the notebooks and results data more portable and easier to run off-site.

Unfortunately the division of pytorch and numpy is extremely messy. numpy and pytorch are both used within compute_metrics, and the results are then in some weird way converted to numpy after in case anything wasn't already numpy.

One problem is that that conversion – the `_to_numpy` function – is not called in collect.py, and sometimes we then get these torch tensors in the results. This effect has even leaked into experiments_lib.py in the analysis part of the repo, which should really not need to deal with this, but there is a comment pointing out that a float cast is because something might be torch.

Now, if you were lazy and completely didn't get my point throughout this whole document, you'd just also call `_to_numpy` in collect.py, but that's obviously just a shitty bandaid to the way that collect.py utilises compute_metrics. If compute_metrics is providing a function, it should provide something complete that the consumer doesn't need to fix itself. To point out something that really points that we need to make the code more intuitive for agents and less easy to break:

We see that there was a step merging feature added to compute_metrics. This was added because Claude didn't know that compute_metrics's functionality was also used by collect.py, since it wasn't in their context. This kind of mistake is really dangerous and is the kind of thing we want to avoid in the future, we see that the saving implementation was implemented twice, and since one saving feature needed to be changed, Claude changed that one saving mechanism, without considering that there were other versions doing the same. We need to somehow find ways to eliminate the possibility of mistakes like this. This current issue happened because the metric compute should generally be in charge of how it's saved, but it's difficult to find the general thing to look out for here. Anything that helps avoid duplication would be welcome.

Going back to the np/pt blurry division however. It's genuinely a hard design decision. I think that if we had to choose one, maybe we should do pytorch only, and then _only_ convert to np before saving, guaranteeing there is no torch format left. We would ideally want to avoid needing to have comments like this:

```python
    # Convert to numpy for numpy-only helper functions
```

But unfortunately, there are functions used that iirc are genuinely specific to numpy, which really complicates things. This makes us do things like convert to numpy before calling, but also converting from pytorch just in case in that function we call. This is something we really don't want. The problem with making it all numpy on the other hand is compatibility with other files and the fact that things will likely again devolve into madness like it is now. So I'm slightly leaning towards pytorch everywhere where possible (type hints must then enforce pytorch, with np native processes owning the responsibility to take and output pytorch). Maybe that's kinda ugly though. Another option is to very clearly put numpy functions in a separately labelled part of the document and make it clear they are exceptions. Idk. This needs discussion.

Also somewhat tangential but still on the topic of compute_metrics, we now have an unused path `compute_derive` that did cpu based derivation. It can probably be removed.


## Code and Code Style

My main value for style is _succinctness_.

Here a quote of the readability section from Paul Graham's essay _Succinctness is Power_:

> We have to be careful here to distinguish between the readability of an individual line of code and the readability of the whole program. It's the second that matters. I agree that a line of Basic is likely to be more readable than a line of Lisp. But a program written in Basic is is going to have more lines than the same program written in Lisp (especially once you cross over into Greenspunland). The total effort of reading the Basic program will surely be greater.
>
> total effort = effort per line x number of lines

And furthermore, I'd like to add that I think there is a constant baseline of effort per line simply due to needing to gloss over it to see what is written. So even very easy lines of code may not even be as easy to read as one thinks.

The code style must be short, this includes comments and docstrings.
Good code is ideally concise and self documenting with no large docstring. Concise code is information dense per token, and if done well, doesn't need much explanation. A text expanation itself also takes up space on screen for humans and takes up more tokens and thus also needs more tokens attended to, making your attention process itself harder.

Example for code:

Claude wrote

```python
def _expand_presets(hooks) -> List[str]:
    out = []
    for entry in hooks or []:
        if entry.startswith("preset:"):
            name = entry[len("preset:"):]
            if name not in _PRESETS:
                raise ValueError(f"unknown preset {name!r}; known: {sorted(_PRESETS)}")
            out.extend(_PRESETS[name])
        else:
            out.append(entry)
    return out
```

Which is 10 lines just to add presets, copying every entry over to a new list one by one in a full-size for loop. That's so silly!

my version:
```python
def _expand_presets(hooks: List[str], prefix: str = "preset:") -> List[str]:
    return [x for h in hooks for x in (_PRESETS[h.removeprefix(prefix)] if h.startswith(prefix) else [h])]
```
One might think to add a try-except here but Python's default error is clear enough, so why bother cluttering up the code.

And in my opinion, this is also much easier to read! It's a single list comprehension and the mechanism – prefix matching and expanding the presets using a dictionary – is directly visible from the variable names and code structure.

Hopefully this serves as an example for how to code. There are many such examples. This is just one of them.
Basically anything that _could_ be a one-liner. Probably should be. And if it gets really long, splitting it up should preferably still follow the declarative style. It's better to extract some parts of this into a function than it is to make a big for loop.

Multi-if blocks are also very unwanted.

Another thing which seems to be everywhere is just random stuff that is completely heterogeneous (i.e. not all cases are treated homogeneously) when it should not be mentioned at all. One example inside DataAccessor:

```python

    # ------------------------------------------------------------------
    # Node / FactorView access points
    # ------------------------------------------------------------------

    @property
    def v(self) -> "Node":
        return Node(self, "")

    def __getitem__(self, path: str) -> "Node":
        return Node(self, path)

    @property
    def after_final_norm(self) -> "Node":
        return Node(self, "after_final_norm")

    @property
    def before_final_norm(self) -> "Node":
        return Node(self, "before_final_norm")

    def factor(self, leaf, q) -> Optional["FactorView"]:
        """Handle-or-None: a FactorView iff (leaf, q) has a reachable representation."""
        return FactorView(self, leaf, q) if self.can_resolve(leaf, q, "eigvals") else None
```

So... we normally use `self.v.<path>` or `self["<path>"]` but for some reason we have hardcoded `after_final_norm` and `before_final_norm`. Why????

Another example that is wrong here is that we have the name FactorView, which is very confusing and must be left over from when things were centered around k-fac, since we are not kfac-centric, we should not be using the term factor anymore (unless it's genuinely referring to the factors of kfac specifically, but that is rarely something that should be treated specially). The FactorView class should really be called Quantity or View, or something else (discuss).

To improve code quality and reduce documentation necessity, I do propose adding typing to every function, any kind of implicit documentation is good, if you have more suggestions, do tell me.


## Agent documentation

The CLAUDE.md is likely outdated and we should spend some time chatting about it. Breaking down every section and rewriting it from the ground up. The CLAUDE.md itself is a thing that fails to be maintained causing confusing and incorrect statements about the codebase, which is why it needs to be looked through by hand. We will go through this together.


## Task

There are many separate passes of the codebase required to really get all of these down. We first need to handle the really structurally deep task, although doing so with consideration for my code quality and style preferences. To do this, I propose we branch after this conversation turn to do the more specific structural changes like the examples I mentioned here (although we'll add in the FactorView things). Then do the real structural audit to find more examples of that and genuinely decouple the codebase, and add some kind of structural way to enforce clear division of tasks and relationships macroscopically; it's this step in which we really fight the cause, not the symptoms. Then after that we return to the branching point of the conversation (I can almost guarantee the context will be otherwise), and first do a deep pass of code quality and style, and effectively minimise and simplify the code anywhere that it is feasible, if it hasn't already been completely rewritten during the first task. Then, finally, we reread the cleaned code, and we go through the documentation together.