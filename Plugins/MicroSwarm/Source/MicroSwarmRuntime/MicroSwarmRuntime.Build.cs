using System.IO;
using UnrealBuildTool;

public class MicroSwarmRuntime : ModuleRules
{
    public MicroSwarmRuntime(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
                "CoreUObject",
                "Engine"
            }
        );

        string RepoRoot = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", "..", "..", ".."));
        string ApiInclude = Path.Combine(RepoRoot, "src");
        PublicIncludePaths.Add(ApiInclude);
    }
}
