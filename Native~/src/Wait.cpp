// Address waits for the solver's job lanes: a lane with nothing to do parks on a 32-bit word until another changes it, which
// gives its core back where spinning would keep it. Burst can only call into the system through a native plugin, so the
// managed side takes these as function pointers and calls them from its jobs.

#include <cstdint>

#include "IUnityInterface.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")
#elif defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#elif defined(__APPLE__)
// The private calls libc++ uses for its atomic waits, present since macOS 10.12.
extern "C" int __ulock_wait(uint32_t operation, void* address, uint64_t value, uint32_t timeoutMicroseconds);
extern "C" int __ulock_wake(uint32_t operation, void* address, uint64_t wakeValue);
static constexpr uint32_t UlCompareAndWait = 1;
static constexpr uint32_t UlfWakeAll = 0x00000100;
#endif

namespace Retooled
{
    // Waits while *word still holds expected, for at most the given time. It may return early, so callers check again.
    static void UNITY_INTERFACE_API WaitOnWord(int32_t* word, int32_t expected, int32_t timeoutMicroseconds)
    {
#if defined(_WIN32)
        DWORD milliseconds = static_cast<DWORD>((timeoutMicroseconds + 999) / 1000);
        WaitOnAddress(word, &expected, sizeof(int32_t), milliseconds);
#elif defined(__linux__)
        timespec timeout{ timeoutMicroseconds / 1000000, (timeoutMicroseconds % 1000000) * 1000L };
        syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0);
#elif defined(__APPLE__)
        __ulock_wait(UlCompareAndWait, word, static_cast<uint32_t>(expected), static_cast<uint32_t>(timeoutMicroseconds));
#endif
    }

    // Wakes one thread waiting on word, or all of them.
    static void UNITY_INTERFACE_API WakeWord(int32_t* word, int32_t all)
    {
#if defined(_WIN32)
        if (all != 0)
        {
            WakeByAddressAll(word);
        }
        else
        {
            WakeByAddressSingle(word);
        }
#elif defined(__linux__)
        syscall(SYS_futex, word, FUTEX_WAKE_PRIVATE, all != 0 ? INT32_MAX : 1, nullptr, nullptr, 0);
#elif defined(__APPLE__)
        __ulock_wake(UlCompareAndWait | (all != 0 ? UlfWakeAll : 0), word, 0);
#endif
    }

    extern "C" UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API RetooledPhysics_WaitFunctions(void** wait, void** wake)
    {
        *wait = reinterpret_cast<void*>(&WaitOnWord);
        *wake = reinterpret_cast<void*>(&WakeWord);
    }
}
