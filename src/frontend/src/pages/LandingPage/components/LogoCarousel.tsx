import { motion } from "framer-motion";
import { useMemo } from "react";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import { SIDEBAR_BUNDLES } from "@/utils/styleUtils";

// CSS keyframes for seamless infinite scroll
const scrollStyles = `
  @keyframes scroll-left {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
  }
  @keyframes scroll-right {
    0% { transform: translateX(-50%); }
    100% { transform: translateX(0); }
  }
`;

interface BundleItem {
  display_name: string;
  name: string;
  icon: string;
}

function LogoCard({ bundle }: { bundle: BundleItem }) {
  return (
    <motion.div
      className="flex-shrink-0 bg-white rounded-2xl border border-gray-200 shadow-sm px-3 py-2 mx-3 flex items-center gap-2 h-14 min-w-[140px] hover:shadow-md transition-shadow"
      whileHover={{ scale: 1.05 }}
    >
      <ForwardedIconComponent
        name={bundle.icon}
        className="h-5 w-5 text-gray-700 flex-shrink-0"
      />
      <span className="text-sm font-medium text-gray-700 whitespace-nowrap">
        {bundle.display_name}
      </span>
    </motion.div>
  );
}

function LogoRow({ bundles, direction = "left" }: { bundles: BundleItem[]; direction?: "left" | "right" }) {
  // Duplicate bundles for seamless loop
  const duplicatedBundles = useMemo(() => [...bundles, ...bundles], [bundles]);

  return (
    <div className="relative overflow-hidden py-1">
      {/* Left fade overlay */}
      <div className="absolute left-0 top-0 bottom-0 w-24 bg-gradient-to-r from-blue-50 via-blue-50/80 to-transparent z-10 pointer-events-none" />
      
      {/* Right fade overlay */}
      <div className="absolute right-0 top-0 bottom-0 w-24 bg-gradient-to-l from-blue-50 via-blue-50/80 to-transparent z-10 pointer-events-none" />
      
      <div
        className="flex"
        style={{
          animation: `scroll-${direction} 30s linear infinite`,
        }}
      >
        {duplicatedBundles.map((bundle, index) => (
          <LogoCard key={`${bundle.name}-${index}`} bundle={bundle} />
        ))}
      </div>
    </div>
  );
}

export default function LogoCarousel() {
  // Filter out generic categories and split bundles into three rows
  const displayBundles = SIDEBAR_BUNDLES.filter(
    (bundle) => 
      bundle.name !== "aiml" && 
      bundle.name !== "languagemodels" && 
      bundle.name !== "embeddings" && 
      bundle.name !== "memories" &&
      bundle.name !== "vectorstores"
  );

  const itemsPerRow = Math.ceil(displayBundles.length / 3);
  const firstRow = displayBundles.slice(0, itemsPerRow);
  const secondRow = displayBundles.slice(itemsPerRow, itemsPerRow * 2);
  const thirdRow = displayBundles.slice(itemsPerRow * 2);

  return (
    <section className="py-20">
      <style dangerouslySetInnerHTML={{ __html: scrollStyles }} />
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
            Seamless Integration
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Browse hundreds of ready-to-use integrations for data sources, models, and vector stores.
          </p>
        </motion.div>

        {/* Carousel Rows */}
        <div className="space-y-1">
          <LogoRow bundles={firstRow} direction="left" />
          <LogoRow bundles={secondRow} direction="right" />
          <LogoRow bundles={thirdRow} direction="left" />
        </div>
      </div>
    </section>
  );
}
