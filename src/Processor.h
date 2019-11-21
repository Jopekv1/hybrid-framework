#pragma once

enum ProcType
{
    cpu,
    gpu
};

struct Processor
{
    const int id;
    const ProcType type;
};