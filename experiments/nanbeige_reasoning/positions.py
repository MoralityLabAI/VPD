"""Token-region selection without model/runtime imports."""


def capture_positions(prompt_length,total_length):
    regions={'prompt':(max(0,prompt_length-8),prompt_length)}
    generated=total_length-prompt_length
    for label,fraction in [('early',.25),('middle',.5),('late',1.)]:
        end=prompt_length+max(1,round(generated*fraction)) if generated else prompt_length
        regions[label]=(max(prompt_length,end-8),min(total_length,end))
    positions=sorted({i for a,b in regions.values() for i in range(a,b)})
    return positions,regions
