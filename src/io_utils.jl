# I/O utilities for SPNNQVI
#
# TeeIO: write to both a file and stdout simultaneously.
# Logging helpers for experiment scripts.

"""
    TeeIO(io1, io2)

An IO wrapper that writes to two streams simultaneously.
"""
struct TeeIO <: IO
    io1::IO
    io2::IO
end

Base.write(tee::TeeIO, x::UInt8) = (write(tee.io1, x); write(tee.io2, x))
Base.write(tee::TeeIO, x::Vector{UInt8}) = (write(tee.io1, x); write(tee.io2, x))
Base.write(tee::TeeIO, x::SubArray{UInt8, 1}) = (write(tee.io1, x); write(tee.io2, x))
Base.flush(tee::TeeIO) = (flush(tee.io1); flush(tee.io2))
Base.close(tee::TeeIO) = (flush(tee.io1); flush(tee.io2))

"""
    setup_logging(experiment_name::String) -> (logpath, tee, logfile)

Create a timestamped log file in results/logs/ and return a TeeIO.
"""
function setup_logging(experiment_name::String)
    logdir = joinpath(@__DIR__, "..", "results", "logs")
    mkpath(logdir)
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMss")
    logpath = joinpath(logdir, "$(experiment_name)_$(timestamp).log")
    logfile = open(logpath, "w")
    tee = TeeIO(stdout, logfile)
    println(tee, "Log: $logpath")
    println(tee, "Started: $(Dates.now())")
    return logpath, tee, logfile
end

"""
    teardown_logging(tee::TeeIO, logpath::String)

Flush and close the log file.
"""
function teardown_logging(tee::TeeIO, logpath::String)
    println(tee, "Finished: $(Dates.now())")
    flush(tee)
    close(tee.io2)  # close the log file
    println("Log saved to: $logpath")
end
