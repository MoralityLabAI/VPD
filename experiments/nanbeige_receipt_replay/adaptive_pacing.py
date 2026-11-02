"""Temperature-dependent idle gaps around the unchanged synchronized decode guard."""
from experiments.nanbeige_reasoning.temperature_guard_burst import TemperatureGuard


class AdaptivePacingGuard(TemperatureGuard):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.policy['idle_temperature_bands']=[
            dict(below_c=74,ratio=.5),dict(below_c=80,ratio=1.),
            dict(below_c=83,ratio=2.),dict(below_c=None,ratio=4.)]
        self.policy['cooling_settle_seconds']=2.

    def _settle_after_cooling(self):
        start=self.clock();stable_since=start
        self.event('cooling_settle_start',temperature_c=self.last_temperature,
                   resume_c=self.policy['resume_c'],stable_seconds=self.policy['cooling_settle_seconds'])
        while stable_since is None or self.clock()-stable_since<self.policy['cooling_settle_seconds']:
            if self.clock()-start>=self.policy['cooling_timeout_seconds']:
                raise RuntimeError('Cooling stability budget exhausted; stopping decoding')
            self.sleep(.25);self.idle_seconds+=.25
            pauses=self.cooling_pauses
            super().check(force=True)
            if self.last_temperature>self.policy['resume_c'] or self.cooling_pauses>pauses:stable_since=None
            elif stable_since is None:stable_since=self.clock()
        self.event('cooling_settle_end',temperature_c=self.last_temperature,paused_seconds=self.clock()-start)

    def check(self,force=False):
        # Parent owns sensing, synchronized pauses, timeout and 87 C hard stop.
        pauses=self.cooling_pauses
        super().check(force=force)
        if self.cooling_pauses>pauses:self._settle_after_cooling()
        if self.last_temperature is None:return
        for band in self.policy['idle_temperature_bands']:
            if band['below_c'] is None or self.last_temperature<band['below_c']:
                ratio=band['ratio'];break
        if self.policy['idle_to_decode_ratio']!=ratio:
            self.policy['idle_to_decode_ratio']=ratio
            self.event('pacing_change',temperature_c=self.last_temperature,idle_to_decode_ratio=ratio)
