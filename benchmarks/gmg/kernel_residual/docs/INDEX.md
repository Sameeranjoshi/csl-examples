# GMG CSL Implementation - Complete Documentation Index

## 🎯 Start Here

**New to GMG V-cycle state machine?** → [README_STATE_MACHINE.md](README_STATE_MACHINE.md)

**Want to run it now?** → Run `./commands_vcycle_wse3.sh`

**Need timing info?** → [QUICK_TIMING_REFERENCE.md](QUICK_TIMING_REFERENCE.md)

## 📚 Documentation Guide

### For Beginners

1. **[README.md](README.md)** - Original GMG overview and concepts
2. **[README_STATE_MACHINE.md](README_STATE_MACHINE.md)** - State machine introduction
3. **[COMPARISON.md](COMPARISON.md)** - Host-controlled vs state machine

### For Implementation Details

4. **[VCYCLE_STATE_MACHINE_SUMMARY.md](VCYCLE_STATE_MACHINE_SUMMARY.md)** - Complete technical docs
5. **[TIMING_GUIDE.md](TIMING_GUIDE.md)** - Timing implementation details
6. **[TIMING_FLOW_DIAGRAM.md](TIMING_FLOW_DIAGRAM.md)** - Visual timing flow

### Quick Reference

7. **[QUICK_TIMING_REFERENCE.md](QUICK_TIMING_REFERENCE.md)** - Timing quick reference
8. **[INDEX.md](INDEX.md)** - This file

## 🗂️ File Organization

### Implementation Files

```
src/
├── kernel_gmg.csl              # Original: host-controlled
├── kernel_gmg_vcycle.csl       # New: state machine with timing ⭐
├── layout_gmg.csl              # Original: host-controlled
├── layout_gmg_vcycle.csl       # New: state machine ⭐
└── blas.csl                    # Shared: BLAS operations
```

### Run Scripts

```
run_gmg.py                      # Original: calls each operation separately
run_gmg_vcycle.py               # New: single kernel call with timing ⭐
```

### Command Scripts

```
commands_wse2.sh                # Original: host-controlled
commands_vcycle_wse3.sh         # New: state machine with timing ⭐
```

### Documentation

```
README.md                       # GMG algorithm overview
README_STATE_MACHINE.md         # State machine intro ⭐
COMPARISON.md                   # Implementation comparison
VCYCLE_STATE_MACHINE_SUMMARY.md # Complete technical docs
TIMING_GUIDE.md                 # Timing implementation
TIMING_FLOW_DIAGRAM.md          # Visual timing flow
QUICK_TIMING_REFERENCE.md       # Quick reference card
INDEX.md                        # This navigation guide
```

## 🎓 Learning Path

### Path 1: I Want to Run GMG

1. Read: [README_STATE_MACHINE.md](README_STATE_MACHINE.md) (5 min)
2. Run: `./commands_vcycle_wse3.sh` (2 min)
3. Understand: [QUICK_TIMING_REFERENCE.md](QUICK_TIMING_REFERENCE.md) (5 min)

**Total time**: ~10-15 minutes

### Path 2: I Want to Understand the Algorithm

1. Read: [README.md](README.md) - GMG basics (10 min)
2. Check: `../python_gmg/gmgoscar.py` - Python reference (10 min)
3. Read: [VCYCLE_STATE_MACHINE_SUMMARY.md](VCYCLE_STATE_MACHINE_SUMMARY.md) (15 min)
4. Study: `src/kernel_gmg_vcycle.csl` - State machine code (20 min)

**Total time**: ~1 hour

### Path 3: I Want to Add Custom Timing

1. Read: [TIMING_GUIDE.md](TIMING_GUIDE.md) - Implementation details (15 min)
2. Study: [TIMING_FLOW_DIAGRAM.md](TIMING_FLOW_DIAGRAM.md) - Visual guide (10 min)
3. Reference: `src/kernel_gmg_vcycle.csl` - Existing timing code (10 min)
4. Implement: Add your custom timing (30 min)

**Total time**: ~1 hour

### Path 4: I Need to Debug

1. Run with debug: `./commands_vcycle_wse3.sh 2>&1 | tee debug.log`
2. Check: `sim.log` for state transitions
3. Verify: State counts match expected (see [VCYCLE_STATE_MACHINE_SUMMARY.md](VCYCLE_STATE_MACHINE_SUMMARY.md))
4. Compare: With Python reference output

## 🔍 Quick Lookup

### "How do I...?"

| Question | Answer |
|----------|--------|
| Run the state machine version? | `./commands_vcycle_wse3.sh` |
| Run the original version? | `./commands_wse2.sh` |
| See timing breakdown? | Check "Performance Timing" in output |
| Change iterations? | `--pre-iter=X --post-iter=Y --bottom-iter=Z` |
| Change grid size? | Recompile with new `width:X,height:Y,MAX_ZDIM:Z,LEVELS:L` |
| Add custom timing? | See [TIMING_GUIDE.md](TIMING_GUIDE.md) section "Custom Timing Regions" |
| Debug state machine? | Check `sim.log` for debug prints |
| Compare with bricks? | See timing output format (similar to bricks) |

### "What file should I read for...?"

| Topic | File |
|-------|------|
| Algorithm overview | [README.md](README.md) |
| State machine intro | [README_STATE_MACHINE.md](README_STATE_MACHINE.md) |
| Why use state machine? | [COMPARISON.md](COMPARISON.md) |
| State machine details | [VCYCLE_STATE_MACHINE_SUMMARY.md](VCYCLE_STATE_MACHINE_SUMMARY.md) |
| Timing implementation | [TIMING_GUIDE.md](TIMING_GUIDE.md) |
| Timing visualization | [TIMING_FLOW_DIAGRAM.md](TIMING_FLOW_DIAGRAM.md) |
| Quick timing reference | [QUICK_TIMING_REFERENCE.md](QUICK_TIMING_REFERENCE.md) |
| Python reference code | `../python_gmg/gmgoscar.py` |
| CSL kernel code | `src/kernel_gmg_vcycle.csl` |

## 📊 Documentation Roadmap

```
┌────────────────────────┐
│  README_STATE_MACHINE  │  ← Start here!
└───────────┬────────────┘
            ↓
    ┌───────┴───────┐
    ↓               ↓
┌─────────────┐  ┌──────────────────────┐
│ COMPARISON  │  │ VCYCLE_STATE_MACHINE │
│             │  │      _SUMMARY        │
└─────────────┘  └──────────┬───────────┘
                            ↓
                 ┌──────────┴───────────┐
                 ↓                      ↓
          ┌──────────────┐      ┌────────────────┐
          │ TIMING_GUIDE │      │ TIMING_FLOW    │
          │              │      │   _DIAGRAM     │
          └──────────────┘      └────────────────┘
                 ↓
          ┌──────────────────────┐
          │ QUICK_TIMING_        │
          │   REFERENCE          │
          └──────────────────────┘
```

## 🎯 Key Features Summary

### State Machine ✨

- **21 states** managing complete V-cycle
- **Automatic sequencing** via callbacks
- **No host intervention** during execution
- **50-100× faster** than host-controlled

### Timing System ⏱️

- **4 operation types** measured
- **Per-level granularity**
- **Min/avg/max statistics**
- **Microsecond resolution**
- **Bricks-compatible** output format

### Verification ✅

- **Matches Python reference**
- **Comprehensive debug prints**
- **State transition tracking**
- **Numerical accuracy checks**

## 💡 Pro Tips

1. **Start small**: Test with 2 levels first, then scale up
2. **Check sim.log**: Verify state transitions before trusting timing
3. **Compare outputs**: Run both host-controlled and state machine, compare timing
4. **Focus on bottlenecks**: Optimize operations with largest time
5. **Use verbose mode**: `--verbose` or `-v` for detailed output

## 🐛 Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All timing = 0 | Timer not enabled | Check `f_enable_timer()` called |
| Wrong results | Data layout mismatch | Verify column-major flattening |
| Hang | State machine bug | Check sim.log for last state |
| Huge times | Infinite loop | Check iteration counters |
| Timing mismatch | Wrong level indexing | Verify `current_level` usage |

## 📞 Support & References

### Internal References

- **PCG Example**: `../preconditioned-conjugate-gradient/run_pcg.py`
- **Power Method**: `../power-method/run_power.py`
- **Timing Example**: `../single-tile-matvec/src/pe_matvec.csl`
- **Python Reference**: `../python_gmg/gmgoscar.py`

### CSL Documentation

- **State machine pattern**: See PCG implementation
- **Timing modules**: `<time>` and `<timer>` documentation
- **Async callbacks**: Stencil and reduce module docs

## 🎉 You're Ready!

You now have:
- ✅ Complete GMG V-cycle state machine
- ✅ Comprehensive performance timing
- ✅ Detailed documentation
- ✅ Working examples
- ✅ Verification against reference

**Next step**: Run `./commands_vcycle_wse3.sh` and see your timing results!

---

**Questions?** Check the relevant doc file from the roadmap above, or grep the codebase for examples!

