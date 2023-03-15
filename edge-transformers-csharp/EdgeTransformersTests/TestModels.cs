using EdgeTransformers.FFI;

namespace EdgeTransformersTests
{
    [TestClass]
    public class UnitTest1
    {
        [TestMethod]
        public void TestModels()
        {
            var env = EnvContainer.New();
            
            var conditionalGen = ConditionalGenerationPipelineFFI.FromPretrained(
                env.Context, "optimum/gpt2", DeviceFFI.CPU, GraphOptimizationLevelFFI.Level3
            );
            var outp = conditionalGen.GenerateTopkSampling("Hello", 2, 50, 0.9f);
            Assert.IsNotNull(outp);
            
            var conditionalGenPkvs = ConditionalGenerationPipelineWithPKVsFFI.FromPretrained(
                env.Context, "optimum/gpt2", DeviceFFI.CPU, GraphOptimizationLevelFFI.Level3
            );
            var outp1 = conditionalGenPkvs.GenerateTopkSampling("Hello", 2, 50, 0.9f);
            Assert.IsNotNull(outp1);

            var emb = EmbeddingPipelineFFI.FromPretrained(
                env.Context, "optimum/all-MiniLM-L6-v2", PoolingStrategyFFI.Mean, DeviceFFI.CPU, GraphOptimizationLevelFFI.Level3
            );
            var emb1 = emb.Embed("There is a dog on the bench");
            Assert.AreEqual(emb1.embedding.Count, 384);
        }
    }
}