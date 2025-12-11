function save_laser_position_in_igor()
    variable xpos, ypos
    
    PTS("CypherMotorPath", "LDX")
    xpos = ReadMotorCount(GTS("CypherMotorPath"))
    
    PTS("CypherMotorPath", "LDY")
    ypos = ReadMotorCount(GTS("CypherMotorPath"))
    
    Make/O root:LDX_pos = {xpos, ypos}
end


function save_focus_positions()
	MoveToSampleFocusPosition()
	Make/O root:sample_pos = GV("FocusMotor")
	MoveToTipFocusPosition()
	Make/O root:tip_pos = GV("FocusMotor")
end

function save_head_position_in_igor()
    variable headpos
    variable focuspos
    
    PTS("CypherMotorPath", "Head")
    headpos = ReadMotorCount(GTS("CypherMotorPath"))
    
    Make/O root:head_pos = headpos
	Make/O root:focus_pos = GV("FocusMotor")
    
end


function save_tune()
    Wave/Z Defl = $""
    Wave/Z Phase = $""
    Wave/Z Amp = $GetDF("Tune")+"Amp"
    Wave Freq = $GetDF("Tune")+"Frequency"
    Duplicate/FREE Freq,TimeWave,Raw
    Ax2Wave(Amp,0,TimeWave)
    Wave/Z Disp = $""
    
    if (IsOnTuneGraph("Thermal;SHOFit;"))
        if (IsOnTuneGraph("Thermal"))
            Wave SrcDefl = $GetDF("Tune")+"TunePSD"
        else
            Wave SrcDefl = $GetDF("Tune")+"TuneSHOFit"
        endif
        Duplicate/FREE SrcDefl,Defl
        LinearInterpBravo(DimSize(Amp,0),Defl,$"")
    endif
    if (IsOnTuneGraph("Phase"))
        Wave Phase = $GetDF("Tune")+"Phase"
    endif
    if (IsOnTuneGraph("Dissipation"))
        Wave Disp = $GetDF("Tune")+"Dissipation"
    endif
    
    ARSaveAsForce(1 | (GV("SaveForce") & 2),"SaveForce","Defl;AmpV;Phase;Freq;Disp;Time;",Raw,Defl,Amp,Phase,Freq,Disp,TimeWave,CustomNote="IsTunePlot:1\r")
    MakePanel("MasterForce")
end