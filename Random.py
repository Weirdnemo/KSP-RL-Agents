import krpc
conn = krpc.connect()
vessel = conn.space_center.active_vessel
vessel.control.throttle = 1.0
vessel.control.activate_next_stage()
