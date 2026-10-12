# Migration provenance

The public repository expresses the final scientific semantics directly in reusable modules; it does not patch private experiment scripts at runtime.

Historical authoritative source fingerprints retained for auditability include:

```text
Experiment-4 Step-06 corrected ablation
40b640d515143056adeb2fadc38c1c1f66d056bb3111a88350a0ed5329756b58

Experiment-4 Step-09 mapped rescue
e410e7812c2a798b7989fd4a13d01d0acba4d8d4f77f4edc79886520f5210154

Experiment-4 Step-07 random-control helper (required externally for exact Gate-2)
7aa5b6aa43aa35c90f977413dd7056e1c00b07f58b02629c4af2b9a5fc9861f8

YOLO Exp15A authoritative engine
9bb687c380bb4818e0581dfeb3fe204c9418196f993a60c2b7469e12d5e33e02

RT-DETR Step-10 NONE->CLEAN-KD bridge fit
7a6331ca6c26526a210fd81c06ba79e35bfe756f0786746cea50a91ae0e107a1

RT-DETR Step-11 native local-utility implementation audit
3630346bfb73e79caba01cf04e93f6d91c153044eb2a1e9c3bbbe881be81f725

RT-DETR Step-12 mapped-complement local-utility confirmation
cf2df9d3f36db5c2580235d99fc289cfa5149c415d1f00a8aaac437b8f2a971a

RT-DETR Step-14 finite-intervention-horizon confirmation
1717d241a8b662e5c528e4e039a10684122f6c4d54dd44e94106868d730e0682

RT-DETR Step-15 task-sensitive geometry source
2a9a566802fd6f932f2623ea4225694a0cb765f0344af8b7aa73cc268a043533

RT-DETR Step-17 orientation confirmation source
c9f4e6c5e1c965f5b4abc9f614f7537c56f9f2b4cefa481e4da964489bc0c997

RT-DETR teacher-canonical coordinate-audit source
12fcc30d847b96716ae56fa1525db51c734a2dd5fec9907e163e17d62738bb66
```

These fingerprints identify historical implementations; public migrated stages do not require the corresponding private files except for the explicitly hash-locked Experiment-4 Step-07 random helper.

## Teacher-canonical constructions

### Flattened Experiment-4 region

```text
N_T = reshape(N, K, 256) @ R_NT
n_T = flatten(N_T)
t   = flatten(T)
alpha = <n_T,t> / <t,t>
e_T = alpha t
c_T = n_T - e_T
```

Ablation maps `e_T` back with `R_NT.T`; rescue maps `c_T` with `R_ST.T`.

### Per-cell TSG direction

```text
n_T,i = n_i @ R_NT
alpha_i = <n_T,i,t_i> / <t_i,t_i>
c_T,i = n_T,i - alpha_i t_i
m_S,i = c_T,i @ R_ST.T
```

The direction tensor contains `m_S,i` at every unique spatial CORE9 support cell simultaneously.

## Historical RT-DETR Step-12/14 construction

The frozen finite-horizon chain is reproduced with its historical Step-10 bridge, not relabeled as a teacher-canonical result:

```text
n_E = (<n,t>/<t,t>) t
n_C = n - n_E
m   = n_C @ R_NONE_TO_CLEAN_KD
```

The later teacher-canonical construction remains authoritative for the final TSG orientation stage. This separation prevents a coordinate-system correction from silently changing the identity of the frozen Step-12/14 experiment.


## Public derivative of the Experiment-4 Step-07 helper

The historical helper remains identified by SHA-256 `7aa5b6aa43aa35c90f977413dd7056e1c00b07f58b02629c4af2b9a5fc9861f8`. Because the historical source contains author-machine absolute path literals, the public bundle stages a path-sanitized derivative only after verifying that historical SHA. The scientific callable `make_random_control_delta` must retain an identical AST (docstring and source-location metadata excluded), and the staged manifest records both the historical source SHA and public derivative SHA.
